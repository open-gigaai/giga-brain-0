# How to contribute to GigaModels?

[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.0-4baaaa.svg)](CODE_OF_CONDUCT.md)

GigaModels is an open source project, so all contributions and suggestions are welcome.

You can contribute in many different ways: giving ideas, answering questions, reporting bugs, proposing enhancements,
improving the documentation, fixing bugs,...

Many thanks in advance to every contributor. In order to facilitate healthy, constructive behavior in an open and inclusive community, we all respect and abide by
our [code of conduct](CODE_OF_CONDUCT.md).

## Code Style

We have some static checks when you commit your code change, please make sure you can pass all the tests and make sure the coding style meets our requirements. Following tools are for linting and formatting:

- [isort](https://github.com/PyCQA/isort): A Python utility to sort imports.
- [black](https://github.com/psf/black): A formatter for Python files.
- [flake8](https://github.com/PyCQA/flake8): A wrapper around some linter tools.
- [codespell](https://github.com/codespell-project/codespell): A Python utility to fix common misspellings in text files.
- [mdformat](https://github.com/executablebooks/mdformat): Mdformat is an opinionated Markdown formatter that can be used to enforce a consistent style in Markdown files.
- [docformatter](https://github.com/myint/docformatter): A formatter to format docstring.

Style configurations of isort, black, flake8 and codespell can be found in [setup.cfg](./setup.cfg).

We use [pre-commit hook](https://pre-commit.com/) that checks and formats.
The config for a pre-commit hook is stored in [.pre-commit-config](./.pre-commit-config.yaml).

After you clone the repository, you will need to install initialize pre-commit hook.

```shell
pip3 install pre-commit
```

## How to work on an open Issue?

You have the list of open Issues at [here](https://github.com/open-gigaai/giga-models/issues).

Some of them may have the label `help wanted`: that means that any contributor is welcomed!

If you would like to work on any of the open Issues:

1. Make sure it is not already assigned to someone else. You have the assignee (if any) on the top of the right column of the Issue page.

2. You can self-assign it by commenting on the Issue page with the keyword: `#self-assign`.

3. Work on your self-assigned issue and eventually create a Pull Request.

## How to create a Pull Request?

You need to follow these steps below to make contribution to the main repository via pull request. You can learn about the details of pull request [here](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests).

1. Fork the [repository](https://github.com/open-gigaai/giga-models) by clicking on the 'Fork' button on the repository's page. This creates a copy of the code under your GitHub user account.

2. Clone your fork to your local disk, and add the base repository as a remote:

   ```bash
   git clone https://github.com/open-gigaai/giga-models.git
   ```

   You need to set the official repository as your upstream so that you can synchronize with the latest update in the official repository. You can learn about upstream [here](https://www.atlassian.com/git/tutorials/git-forks-and-upstreams).

   ```shell
   cd giga-models
   git remote add upstream https://github.com/open-gigaai/giga-models.git
   ```

   you can use the following command to verify that the remote is set. You should see both `origin` and `upstream` in the output.

   ```shell
   git remote -v
   ```

3. Create a new branch to hold your development changes:

   ```bash
   git checkout -b a-descriptive-name-for-my-changes
   ```

   **do not** work on the `main` branch.

4. Develop the features on your branch

5. (Optional) You can use [`pre-commit`](https://pre-commit.com/) to format your code automatically each time run `git commit`.
   To do this, install `pre-commit` via `pip3 install pre-commit` and then run `pre-commit install` in the project's root directory to set up the hooks.
   Before you create a PR, make sure that your code lints and is formatted by black.

   ```shell
   pre-commit run --all-files
   ```

   Note that if any files were formatted by `pre-commit` hooks during committing, you have to run `git commit` again.

6. Once you're happy with your contribution, add your changed files and make a commit to record your changes locally:

   ```bash
   git add -u
   git commit
   ```

   It is a good idea to sync your copy of the code with the original
   repository regularly. This way you can quickly account for changes:

   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

7. Once you are satisfied, push the changes to your fork repo using:

   ```bash
   git push -u origin a-descriptive-name-for-my-changes
   ```

   Go the webpage of your fork on GitHub. Click on "Pull request" to send your changes to the project maintainers for review.
