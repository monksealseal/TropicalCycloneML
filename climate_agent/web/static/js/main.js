/**
 * Climate Insight - Landing Page JavaScript
 */

document.addEventListener('DOMContentLoaded', () => {
    // Waitlist form submission
    const form = document.getElementById('waitlist-form');
    const success = document.getElementById('waitlist-success');

    if (form) {
        form.addEventListener('submit', async (e) => {
            e.preventDefault();

            const email = document.getElementById('email').value;
            const company = document.getElementById('company').value;
            const useCase = document.getElementById('use_case').value;

            const btn = form.querySelector('button[type="submit"]');
            const originalText = btn.textContent;
            btn.textContent = 'Joining...';
            btn.disabled = true;

            try {
                const resp = await fetch('/api/v1/waitlist', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        email: email,
                        company: company || null,
                        use_case: useCase || null,
                    }),
                });

                const data = await resp.json();

                form.style.display = 'none';
                success.style.display = 'block';
            } catch (err) {
                btn.textContent = 'Try Again';
                btn.disabled = false;
                console.error('Waitlist error:', err);
            }
        });
    }

    // Scroll to pricing if URL has #pricing or scroll_to param
    if (window.location.hash === '#pricing') {
        setTimeout(() => {
            document.getElementById('pricing')?.scrollIntoView({ behavior: 'smooth' });
        }, 100);
    }

    // Smooth nav background on scroll
    const nav = document.querySelector('.nav');
    window.addEventListener('scroll', () => {
        if (window.scrollY > 50) {
            nav.style.background = 'rgba(10, 15, 28, 0.95)';
        } else {
            nav.style.background = 'rgba(10, 15, 28, 0.85)';
        }
    });
});
