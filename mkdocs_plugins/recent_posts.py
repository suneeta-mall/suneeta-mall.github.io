import re
from datetime import datetime
from pathlib import Path

from mkdocs.plugins import BasePlugin


class RecentPostsPlugin(BasePlugin):
    def __init__(self: "RecentPostsPlugin") -> None:
        self.recent_posts = []

    def on_files(
        self: "RecentPostsPlugin", files: list[dict], config: dict
    ) -> list[dict]:
        # Get all blog post files
        blog_posts = []
        for file in files:
            if file.src_path.startswith("blog/posts/") and file.src_path.endswith(
                ".md"
            ):
                # Extract date from filename (format: YYYY-MM-DD-title.md)
                match = re.match(
                    r"blog/posts/(\d{4})-(\d{2})-(\d{2})-.*\.md", file.src_path
                )
                if match:
                    year, month, day = match.groups()
                    try:
                        date = datetime.strptime(f"{year}-{month}-{day}", "%Y-%m-%d")  # noqa: DTZ007
                        with Path(file.abs_src_path).open("r", encoding="utf-8") as f:
                            content = f.read()
                            title_match = re.search(
                                r'title:\s*["\']?(.+?)["\']?(?:\n|$)', content
                            )
                            if title_match:
                                title = title_match.group(1).strip()
                                slug = title.lower()
                                slug = re.sub(r"[^\w\s-]", "", slug)
                                slug = re.sub(r"\s+", "-", slug)
                                slug = re.sub(r"-+", "-", slug)
                                slug = slug.strip("-")
                                url_path = f"/blog/{year}/{month}/{day}/{slug}/"

                                blog_posts.append(
                                    {"date": date, "title": title, "url": url_path}
                                )
                    except ValueError:
                        continue

        self.recent_posts = sorted(blog_posts, key=lambda x: x["date"], reverse=True)[
            :3
        ]
        return files

    def on_page_markdown(
        self: "RecentPostsPlugin",
        markdown: str,
        page: dict,
        config: dict,
        files: list[dict],
    ) -> str:
        if page.file.src_path == "README.md":
            # Create the recent posts section
            recent_posts_md = '## Recent Blog Posts\n\n!!! abstract "Latest Articles"\n'
            for post in self.recent_posts:
                date_str = post["date"].strftime("%B %d, %Y")
                recent_posts_md += (
                    f"    - [{date_str} - {post['title']}]({post['url']})\n"
                )

            # Replace the existing recent posts section
            markdown = re.sub(
                r"## Recent Blog Posts.*?(?=## |$)",
                recent_posts_md,
                markdown,
                flags=re.DOTALL,
            )
        return markdown
