# Econometrics II (2025)
Welcome! This is the repo where you'll find all the material seen in class.
It is also the place you'll be pushing all your assignments to.

This repo is meant to resemble a typical industry-grade project. This means
you'll have to pass a few tests for your code to be accepted.

## Why this workflow?
In this class, you're expected to submit your code by creating a new branch,
pushing your changes and passing a few builds. This will enable you to merge
your branch into `develop`, which will later be merged on to `main`. The goal of
this workflow is to help you understand how code is written and shared in modern
companies. It will familiarize you with:
- Writing clean, consistent, and well-documented code;
- Opening pull requests (PRs) to submit your work for review; and
- Passing builds.

By following these guidelines, you won't be taken by surprise when someone
asks you to open a PR to squash a bug or implement a new feature.

## Structure of this repo
```
.
├── .github           # Build configuration (ignore this)
├── assignments       # Model assignments
│   ├── did    
│   ├── fes
│   ├── ivs
│   ├── rct
│   └── rdd
├── lecture-notes     # Lecture notes
├── .flake8           # Linting configuration (ignore this)
├── .gitignore        # Files ignored by git (ignore this)
├── environment.yml   # Use this to create your environment
├── README.md         # Intro to this repo
└── requirements.txt  # Development environment (ignore this)
```

## Contributing (AKA submitting your homework)
1. Clone the repo to your local (if you haven't done so already)
3. Switch to `develop` (`git checkout develop`)
2. Pull the latest changes (`git pull origin develop`)
3. Create a branch (`git checkout -b assignment/{homework id}-{student id}`)
    - For example `git checkout -b assignment/ivs-659402`
4. Submit a single Python file
    - Must be named `{homework id}-{student id}.py` (eg, `ivs-659402.py`)
    - Must be placed in the corresponding homework's `code` directory (eg,
    `./assignments/ivs/code/`)
    - Submit the file by addig, commiting and pushing your code
5. If your branch and Python script are named correctly (steps 3 and 4), your branch
will trigger a few [Actions](https://github.com/ArturoSbr/econometrics-ii-2025) and
grade your code.
