# Econometrics II Final Project

This final project requires you to build a **fully-documented**, **reproducible**, and
**well-organized** data scraping pipeline. You will be working with the public API of
your choice (though I recommend using the YouTube Data API since we used it in class).

## Project Requirements

### 1. GitHub Repository (25 points)

Your submission must be a **public GitHub repository**. It must include:

- `README.md`  
  - Clearly states the purpose of the project and how to use it.

- `.gitignore`  
  - Must ignore secret files like `.env`, credentials, or other private info.

- Repo structure:
  ```
  your-repo/
  ├── README.md
  ├── .gitignore
  ├── requirements.txt OR environment.yml
  ├── code/
  │   └── scrape_comments.py    # Python scraper
  └── data/
      └── dataset.csv           # Final dataset
  ```

### 2. Reproducible Environment (25 points)

Your project must have a reproducible environment. Include one of the following in the
root of your repo:

- `requirements.txt` for `venv`
- `environment.yml` for `conda`  

Your environment instructions must only contain explicitly used libraries with pinned
versions. For example:

- `python-dotenv==1.0.1` (Good!)
- `python-dotenv` (Bad)

Full points will be awarded based on:

1. Environment instructions can be run;
2. There are no transitive dependencies (library requirements such as `cython`), only
explicitly-used libraries; and
3. Library versions are pinned.

### 3. Scraping Script (25 points)

- Your script should use an API (e.g., YouTube Data API v3).
- Save your code in the `code/` directory.
- Load API keys secretly using `dotenv` or another safe method.
- Must be PEP-8 compliant

### 4. Final Dataset (25 points)

- Save your final dataset as a `.csv` file in the `data/` folder.
- Your dataset must:
  - Contain **at least 500 rows**
  - Include a column that identifies **where the data came from**  
    (e.g., video title, video URL, subreddit name, etc.)

Example format:

| comment_id | text       | video_id        |
|------------|------------|-----------------|
| abc123     | Nice vid!  | acCdeFghiJk     |
| xyz789     | Cool beans | acCdeFghiJk     |

## Submission Instructions

- Make sure your GitHub repo is **public** and complete
- Submit the URL of your repository to the course portal
- Deadline: 2025-05-31