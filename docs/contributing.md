# Join us in making a difference!

Your contributions are always welcome and we would love to see how you can make our project even better. Your input is invaluable to us, and we ensure that all contributors receive recognition for their efforts.

## Ways to contribute

Here’s how you can get involved:

### Report Bugs

Report bugs at <https://github.com/Blaizzy/fastmlx/issues>.

If you are reporting a bug, please include:

-   Your operating system name and version.
-   Any details about your local setup that might be helpful in troubleshooting.
-   Detailed steps to reproduce the bug.

### Fix Bugs

Look through the GitHub issues for bugs. Anything tagged with `bug` and `help wanted` is open to whoever wants to implement it.

### Implement Features

Look through the GitHub issues for features. If anything tagged `enhancement` and `help wanted` catches your eye, dive in and start coding. Your ideas can become a reality in FastMLX!

### Write Documentation

We’re always in need of more documentation, whether it’s for our official docs, adding helpful comments in the code, or writing blog posts and articles. Clear and comprehensive documentation empowers the community, and your contributions are crucial!

### Submit Feedback

The best way to share your thoughts is by filing an issue on our GitHub page: <https://github.com/Blaizzy/fastmlx/issues>. Whether you’re suggesting a new feature or sharing your experience, we want to hear from you!

Proposing a feature?

-    Describe in detail how it should work.
-    Keep it focused and manageable to make implementation smoother.
-    Remember, this is a volunteer-driven project, and your contributions are always appreciated!

## How to get Started!

Ready to contribute? Follow these simple steps to set up FastMLX for local development and start making a difference.

1.  Fork the repository.
    -    Head over to the [fastmlx GitHub repo](<https://github.com/Blaizzy/fastmlx/>) and click the Fork button to create your copy of the repository.

2.  Clone your fork locally
    -    Open your terminal and run the following command to clone your forked repository:

    ```shell
    $ git clone git@github.com:your_name_here/fastmlx.git
    ```

3.  Set Up Your Development Environment
    -    Install your local copy of FastMLX into a virtual environment. If you’re using `virtualenvwrapper`, follow these steps:
    
    ```shell
    $ mkvirtualenv fastmlx
    $ cd fastmlx/
    $ python setup.py develop
    ```

    Tip: If you don’t have `virtualenvwrapper` installed, you can install it with `pip install virtualenvwrapper`.

4.  Create a Development Branch
    -    Create a new branch to work on your bugfix or feature:

    ```shell
    $ git checkout -b name-of-your-bugfix-or-feature
    ```

    Now you’re ready to make changes! 

5.  Run Tests and Code Checks

    -    When you're done making changes, check that your changes pass flake8
    and the tests, including testing other Python versions with tox:

    ```shell
    $ flake8 fastmlx tests
    $ pytest .
    ```

    -    To install flake8 and tox, simply run:
    ```
    pip install flake8 tox
    ```

6.  Commit and Push Your Changes
    -    Once everything looks good, commit your changes with a descriptive message:

    ```shell
    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature
    ```

7.  Submit a Pull Request
    -    Head back to the FastMLX GitHub repo and open a pull request. We’ll review your changes, provide feedback, and merge them once everything is ready.

## Pull Request Guidelines

Before you submit a pull request, check that it meets these guidelines:

1.  The pull request should include tests.
2.  If the pull request adds functionality, the docs should be updated.
    Put your new functionality into a function with a docstring, and add
    the feature to the list in README.rst.
3.  The pull request should work for Python 3.8 and later, and
    for PyPy. Check <https://github.com/Blaizzy/fastmlx/pull_requests> and make sure that the tests pass for all
    supported Python versions.
