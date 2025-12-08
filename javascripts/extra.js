// Back to Top button
document.addEventListener('DOMContentLoaded', function () {
    const backToTop = document.createElement('button');
    backToTop.innerHTML = '↑';
    backToTop.className = 'md-top';
    backToTop.style.cssText = `
        position: fixed;
        bottom: 20px;
        right: 20px;
        display: none;
        width: 40px;
        height: 40px;
        border-radius: 50%;
        background: var(--md-primary-fg-color);
        color: var(--md-primary-bg-color);
        border: none;
        cursor: pointer;
        font-size: 20px;
        z-index: 1000;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    `;
    document.body.appendChild(backToTop);

    window.addEventListener('scroll', function () {
        if (window.pageYOffset > 100) {
            backToTop.style.display = 'block';
        } else {
            backToTop.style.display = 'none';
        }
    });

    backToTop.addEventListener('click', function () {
        window.scrollTo({ top: 0, behavior: 'smooth' });
    });
});

// Share buttons for sections
document.addEventListener('DOMContentLoaded', function () {
    const headings = document.querySelectorAll('h1, h2, h3');
    headings.forEach(heading => {
        const shareButton = document.createElement('button');
        shareButton.innerHTML = '🔗';
        shareButton.className = 'md-share';
        shareButton.style.cssText = `
            background: none;
            border: none;
            cursor: pointer;
            font-size: 1em;
            margin-left: 10px;
            opacity: 0.5;
            transition: opacity 0.2s;
        `;
        shareButton.title = 'Share this section';

        heading.style.display = 'flex';
        heading.style.alignItems = 'center';
        heading.appendChild(shareButton);

        shareButton.addEventListener('click', function (e) {
            e.preventDefault();
            const url = window.location.href.split('#')[0] + '#' + heading.id;
            navigator.clipboard.writeText(url).then(() => {
                const originalText = shareButton.innerHTML;
                shareButton.innerHTML = '✓';
                setTimeout(() => {
                    shareButton.innerHTML = originalText;
                }, 2000);
            });
        });

        heading.addEventListener('mouseenter', () => {
            shareButton.style.opacity = '1';
        });
        heading.addEventListener('mouseleave', () => {
            shareButton.style.opacity = '0.5';
        });
    });
}); 