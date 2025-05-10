import os
import re
from datetime import datetime
from urllib.parse import quote

from mkdocs.plugins import BasePlugin
from mkdocs.structure.files import get_files


class RecentPostsPlugin(BasePlugin):
    def __init__(self):
        self.recent_posts = []

    def on_files(self, files, config):
        # Get all blog post files
        blog_posts = []
        for file in files:
            if file.src_path.startswith('blog/posts/') and file.src_path.endswith('.md'):
                # Extract date from filename (format: YYYY-MM-DD-title.md)
                match = re.match(r'blog/posts/(\d{4})-(\d{2})-(\d{2})-.*\.md', file.src_path)
                if match:
                    year, month, day = match.groups()
                    try:
                        date = datetime.strptime(f"{year}-{month}-{day}", '%Y-%m-%d')
                        # Read the file to get the title from frontmatter
                        with open(file.abs_src_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            # Look for title in frontmatter, handling various quote styles and special characters
                            title_match = re.search(r'title:\s*["\']?(.+?)["\']?(?:\n|$)', content)
                            if title_match:
                                title = title_match.group(1).strip()
                                # Create URL-friendly slug from title
                                slug = title.lower()
                                # Remove quotes and special characters, but preserve hyphens
                                slug = re.sub(r'[^\w\s-]', '', slug)
                                # Replace spaces with hyphens
                                slug = re.sub(r'\s+', '-', slug)
                                # Replace multiple hyphens with single hyphen
                                slug = re.sub(r'-+', '-', slug)
                                # Remove leading/trailing hyphens
                                slug = slug.strip('-')
                                # Construct the URL path
                                url_path = f"/blog/{year}/{month}/{day}/{slug}/"
                                
                                blog_posts.append({
                                    'date': date,
                                    'title': title,
                                    'url': url_path
                                })
                    except ValueError:
                        continue

        # Sort by date (newest first) and take the latest 3
        self.recent_posts = sorted(blog_posts, key=lambda x: x['date'], reverse=True)[:3]
        return files

    def on_page_markdown(self, markdown, page, config, files):
        if page.file.src_path == 'README.md':
            # Create the recent posts section
            recent_posts_md = "## Recent Blog Posts\n\n!!! abstract \"Latest Articles\"\n"
            for post in self.recent_posts:
                date_str = post['date'].strftime('%B %d, %Y')
                recent_posts_md += f"    - [{date_str} - {post['title']}]({post['url']})\n"
            
            # Replace the existing recent posts section
            markdown = re.sub(
                r'## Recent Blog Posts.*?(?=## |$)',
                recent_posts_md,
                markdown,
                flags=re.DOTALL
            )
        return markdown 