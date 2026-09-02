"""Run the GigaBrain policy behind the Simulation Challenge reverse tunnel."""

from __future__ import annotations

import argparse
import asyncio
import enum
import json
import logging
import os
import struct
import threading
import traceback
import uuid
from collections.abc import Callable, Mapping
from typing import Any
from urllib.parse import urlencode, urlparse, urlunparse

import msgpack
import numpy as np
import websockets

try:
    from websockets.asyncio.client import connect as websocket_connect

    _HEADER_ARGUMENT = 'additional_headers'
except ImportError:
    from websockets import connect as websocket_connect

    _HEADER_ARGUMENT = 'extra_headers'

from robocoliseum_challenge_policy import GigaBrainChallengePolicy


LOGGER = logging.getLogger('robocoliseum_challenge_agent')
MAX_SESSION_ID_LENGTH = 256


def encode_data_frame(session_id: str, payload: bytes) -> bytes:
    """Add the gateway's length-prefixed session ID to a payload."""
    encoded_session_id = session_id.encode('utf-8')
    if len(encoded_session_id) > MAX_SESSION_ID_LENGTH:
        raise ValueError(
            f'session_id exceeds {MAX_SESSION_ID_LENGTH} bytes: '
            f'{len(encoded_session_id)}'
        )
    return struct.pack('>I', len(encoded_session_id)) + encoded_session_id + payload


def decode_data_frame(frame: bytes) -> tuple[str, bytes]:
    """Split a gateway data frame into its session ID and payload."""
    if len(frame) < 4:
        raise ValueError('data frame is shorter than its length prefix')
    (session_id_length,) = struct.unpack('>I', frame[:4])
    if session_id_length > len(frame) - 4:
        raise ValueError('session ID length exceeds the available frame bytes')
    session_id_end = 4 + session_id_length
    session_id = frame[4:session_id_end].decode('utf-8')
    return session_id, bytes(frame[session_id_end:])


def _unpack_numpy(value: dict[Any, Any]) -> Any:
    """Decode the safe NumPy representation used by openpi-client msgpack."""
    if b'__ndarray__' in value:
        return np.ndarray(
            buffer=value[b'data'],
            dtype=np.dtype(value[b'dtype']),
            shape=value[b'shape'],
        )
    if b'__npgeneric__' in value:
        return np.dtype(value[b'dtype']).type(value[b'data'])
    return value


def unpack_request(payload: bytes) -> Mapping[str, Any]:
    """Decode and validate one JSON-RPC inference request."""
    request = msgpack.unpackb(payload, object_hook=_unpack_numpy, raw=False)
    if not isinstance(request, Mapping):
        raise ValueError('inference request must be a mapping')
    if request.get('method') != 'infer':
        raise ValueError(f"unsupported request method: {request.get('method')!r}")
    params = request.get('params')
    if not isinstance(params, Mapping):
        raise ValueError('inference request params must be a mapping')
    return params


class PolicyHandler:
    """Load one policy per process and serialize access to its temporal state."""

    def __init__(self, args: argparse.Namespace) -> None:
        self._args = args
        self._policy: GigaBrainChallengePolicy | None = None
        self._lock = threading.Lock()

    def load(self) -> None:
        """Load the checkpoint once during the gateway warmup phase."""
        if self._policy is not None:
            return
        LOGGER.info('loading model from %s', self._args.model_path)
        self._policy = GigaBrainChallengePolicy(
            model_path=self._args.model_path,
            norm_stats_path=self._args.norm_stats_path,
            tokenizer_model_path=self._args.tokenizer_model_path,
            fast_tokenizer_path=self._args.fast_tokenizer_path,
            device=self._args.device,
        )
        LOGGER.info('model loaded')

    def __call__(self, session_id: str, payload: bytes) -> bytes:
        """Handle warmup or run one synchronous model inference call."""
        with self._lock:
            self.load()
            if not payload:
                return b''
            assert self._policy is not None
            params = unpack_request(payload)
            response = self._policy.infer(params)
            LOGGER.info(
                'session=%s episode=%s task=%s',
                session_id,
                params.get('episode_idx'),
                params.get('task_name'),
            )
            return msgpack.packb(response, use_bin_type=True)

    def close_session(self, session_id: str) -> None:
        """Reset model memory when the simulator closes an episode session."""
        with self._lock:
            if self._policy is not None:
                self._policy.reset()
        LOGGER.info('session closed and policy reset: %s', session_id)


class State(str, enum.Enum):
    """Lifecycle states reported by the challenge gateway."""

    QUEUED = 'QUEUED'
    WARMUP = 'WARMUP'
    RUNNING = 'RUNNING'
    DRAINING = 'DRAINING'


FrameHandler = Callable[[str, bytes], bytes]


class TunnelExhausted(RuntimeError):
    """Raised when the tunnel cannot reconnect within its retry budget."""


class TunnelClient:
    """Maintain one reverse WebSocket tunnel to the challenge gateway."""

    def __init__(
        self,
        *,
        url: str,
        access_token: str,
        job_uuid: str,
        frame_handler: FrameHandler,
        session_close_handler: Callable[[str], None],
        agent_id: str | None = None,
    ) -> None:
        self._url = url
        self._access_token = access_token
        self._job_uuid = job_uuid
        self._frame_handler = frame_handler
        self._session_close_handler = session_close_handler
        self.agent_id = agent_id or str(uuid.uuid4())
        self.state = State.QUEUED
        self._drained = False
        self._inflight: set[asyncio.Task[Any]] = set()

    def _connection_url(self) -> str:
        parts = urlparse(self._url)
        query = urlencode({'job': self._job_uuid, 'agent': self.agent_id})
        if parts.query:
            query = f'{parts.query}&{query}'
        return urlunparse(parts._replace(query=query))

    def _set_state(self, state: State) -> None:
        self.state = state
        LOGGER.info('state -> %s', state.value)

    async def _wait_for_inflight(self) -> None:
        """Wait until every response already being computed has been sent."""
        while self._inflight:
            tasks = tuple(self._inflight)
            LOGGER.info('waiting for %d in-flight response(s)', len(tasks))
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _invoke_handler(self, session_id: str, payload: bytes) -> bytes:
        return await asyncio.to_thread(self._frame_handler, session_id, payload)

    async def run(self, max_retries: int) -> None:
        """Reconnect with bounded exponential backoff until the gateway drains."""
        attempt = 0
        while True:
            self._drained = False
            try:
                await self._run_once()
            except (OSError, websockets.exceptions.WebSocketException) as error:
                last_error: Exception = error
                LOGGER.warning('tunnel error: %s', error)
            else:
                if self._drained:
                    return
                last_error = RuntimeError('WebSocket closed without drain')
            if attempt >= max_retries:
                raise TunnelExhausted(
                    f'giving up after {attempt + 1} attempts: {last_error}'
                ) from last_error
            backoff = min(30.0, 0.5 * (2**attempt))
            LOGGER.info('reconnecting in %.1f seconds', backoff)
            await asyncio.sleep(backoff)
            attempt += 1

    async def _run_once(self) -> None:
        connection_url = self._connection_url()
        LOGGER.info('dialing gateway for job=%s agent=%s', self._job_uuid, self.agent_id)
        header_arguments = {
            _HEADER_ARGUMENT: {'Authorization': f'Bearer {self._access_token}'}
        }
        async with websocket_connect(
            connection_url,
            ping_interval=20,
            ping_timeout=10,
            max_size=None,
            **header_arguments,
        ) as websocket:
            async for message in websocket:
                if isinstance(message, str):
                    await self._handle_control(websocket, message)
                    if self.state == State.DRAINING:
                        await self._wait_for_inflight()
                        await websocket.close()
                        return
                elif isinstance(message, (bytes, bytearray)):
                    task = asyncio.create_task(
                        self._handle_data(websocket, bytes(message))
                    )
                    self._inflight.add(task)
                    task.add_done_callback(self._inflight.discard)

    async def _handle_control(self, websocket: Any, message: str) -> None:
        try:
            frame = json.loads(message)
        except json.JSONDecodeError:
            LOGGER.warning('ignoring malformed control frame')
            return
        control_type = frame.get('type')
        session_id = frame.get('session_id', '')
        if control_type == 'warmup':
            self._set_state(State.WARMUP)
            await self._invoke_handler('', b'')
            await websocket.send(json.dumps({'type': 'ready'}))
            self._set_state(State.RUNNING)
        elif control_type == 'session_close' and session_id:
            await asyncio.to_thread(self._session_close_handler, session_id)
        elif control_type == 'drain':
            self._drained = True
            self._set_state(State.DRAINING)

    async def _handle_data(self, websocket: Any, frame: bytes) -> None:
        try:
            session_id, payload = decode_data_frame(frame)
            response = await self._invoke_handler(session_id, payload)
            await websocket.send(encode_data_frame(session_id, response))
        except websockets.exceptions.ConnectionClosed:
            return
        except Exception:
            LOGGER.error('failed to handle data frame\n%s', traceback.format_exc())


def parse_args() -> argparse.Namespace:
    """Parse the explicit runtime paths supplied by the launch script."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--access-token', default=os.environ.get('CHALLENGE_TOKEN', ''))
    parser.add_argument('--job-uuid', required=True)
    parser.add_argument('--gateway-url', required=True)
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--norm-stats-path', required=True)
    parser.add_argument('--tokenizer-model-path', required=True)
    parser.add_argument('--fast-tokenizer-path', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--agent-id')
    parser.add_argument('--max-retries', type=int, default=5)
    args = parser.parse_args()
    if not args.access_token:
        parser.error('--access-token or CHALLENGE_TOKEN is required')
    return args


def main() -> int:
    """Load configuration and run until the gateway sends a drain frame."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(name)s %(levelname)s %(message)s',
        force=True,
    )
    args = parse_args()
    handler = PolicyHandler(args)
    client = TunnelClient(
        url=args.gateway_url,
        access_token=args.access_token,
        job_uuid=args.job_uuid,
        frame_handler=handler,
        session_close_handler=handler.close_session,
        agent_id=args.agent_id,
    )
    try:
        asyncio.run(client.run(args.max_retries))
    except KeyboardInterrupt:
        return 130
    except TunnelExhausted as error:
        LOGGER.error('%s', error)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
