from setuptools import find_packages, setup

setup(
    name="mkdocs-recent-posts-plugin",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "mkdocs>=1.0.0",
    ],
    entry_points={
        "mkdocs.plugins": [
            "recent_posts = mkdocs_plugins.recent_posts:RecentPostsPlugin",
        ]
    },
)
