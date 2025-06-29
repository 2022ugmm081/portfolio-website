// Main JavaScript file for portfolio website
document.addEventListener('DOMContentLoaded', function() {
    // Initialize all components
    initNavigation();
    initScrollAnimations();
    initSkillBars();
    initContactForm();
    initSmoothScroll();
    initTypingEffect();
});

// Navigation functionality
function initNavigation() {
    const navbar = document.querySelector('.navbar');
    const navLinks = document.querySelectorAll('.nav-link');
    
    // Handle navbar background on scroll
    window.addEventListener('scroll', function() {
        if (window.scrollY > 50) {
            navbar.classList.add('scrolled');
        } else {
            navbar.classList.remove('scrolled');
        }
        
        // Update active nav link based on scroll position
        updateActiveNavLink();
    });
    
    // Handle mobile menu
    const navbarToggler = document.querySelector('.navbar-toggler');
    const navbarCollapse = document.querySelector('.navbar-collapse');
    
    if (navbarToggler) {
        navbarToggler.addEventListener('click', function() {
            navbarCollapse.classList.toggle('show');
        });
    }
    
    // Close mobile menu when clicking nav links
    navLinks.forEach(link => {
        link.addEventListener('click', function() {
            if (navbarCollapse.classList.contains('show')) {
                navbarCollapse.classList.remove('show');
            }
        });
    });
}

// Update active navigation link based on scroll position
function updateActiveNavLink() {
    const sections = document.querySelectorAll('section[id]');
    const navLinks = document.querySelectorAll('.nav-link');
    
    let currentSection = '';
    const scrollPosition = window.scrollY + 100;
    
    sections.forEach(section => {
        const sectionTop = section.offsetTop;
        const sectionHeight = section.offsetHeight;
        
        if (scrollPosition >= sectionTop && scrollPosition < sectionTop + sectionHeight) {
            currentSection = section.getAttribute('id');
        }
    });
    
    navLinks.forEach(link => {
        link.classList.remove('active');
        if (link.getAttribute('href') === `#${currentSection}`) {
            link.classList.add('active');
        }
    });
}

// Smooth scrolling for navigation links
function initSmoothScroll() {
    const navLinks = document.querySelectorAll('a[href^="#"]');
    
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            
            const targetId = this.getAttribute('href');
            const targetSection = document.querySelector(targetId);
            
            if (targetSection) {
                const offsetTop = targetSection.offsetTop - 80; // Account for fixed navbar
                
                window.scrollTo({
                    top: offsetTop,
                    behavior: 'smooth'
                });
            }
        });
    });
}

// Scroll animations
function initScrollAnimations() {
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };
    
    const observer = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('animate-fade-in-up');
                
                // Trigger skill bar animations
                if (entry.target.classList.contains('skills-overview')) {
                    animateSkillBars();
                }
                
                // Trigger counter animations
                if (entry.target.classList.contains('hero-metrics')) {
                    animateCounters();
                }
            }
        });
    }, observerOptions);
    
    // Observe elements for animations
    const animatedElements = document.querySelectorAll(
        '.project-card, .timeline-item, .skills-category, .contact-item, .skills-overview, .hero-metrics'
    );
    
    animatedElements.forEach(element => {
        observer.observe(element);
    });
}

// Animate skill bars
function initSkillBars() {
    const skillBars = document.querySelectorAll('.skill-progress');
    
    skillBars.forEach(bar => {
        const width = bar.style.width;
        bar.style.width = '0%';
        bar.dataset.width = width;
    });
}

function animateSkillBars() {
    const skillBars = document.querySelectorAll('.skill-progress');
    
    skillBars.forEach((bar, index) => {
        setTimeout(() => {
            bar.style.width = bar.dataset.width;
        }, index * 200);
    });
}

// Animate counters in hero section
function animateCounters() {
    const counters = document.querySelectorAll('.metric-value');
    
    counters.forEach(counter => {
        const target = counter.textContent;
        const isDecimal = target.includes('.');
        const isRatio = target.includes('/');
        
        if (isDecimal) {
            animateDecimalCounter(counter, parseFloat(target));
        } else if (isRatio) {
            animateRatioCounter(counter, target);
        } else {
            animatePercentageCounter(counter, target);
        }
    });
}

function animateDecimalCounter(element, target) {
    let current = 0;
    const increment = target / 100;
    const timer = setInterval(() => {
        current += increment;
        if (current >= target) {
            current = target;
            clearInterval(timer);
        }
        element.textContent = current.toFixed(2);
    }, 20);
}

function animateRatioCounter(element, target) {
    const [numerator, denominator] = target.split('/').map(num => parseInt(num));
    let current = 0;
    const increment = numerator / 100;
    const timer = setInterval(() => {
        current += increment;
        if (current >= numerator) {
            current = numerator;
            clearInterval(timer);
        }
        element.textContent = `${Math.floor(current)}/${denominator}`;
    }, 20);
}

function animatePercentageCounter(element, target) {
    const numericTarget = parseFloat(target.replace('%', ''));
    let current = 0;
    const increment = numericTarget / 100;
    const timer = setInterval(() => {
        current += increment;
        if (current >= numericTarget) {
            current = numericTarget;
            clearInterval(timer);
        }
        element.textContent = current.toFixed(2) + '%';
    }, 20);
}

// Typing effect for hero subtitle
function initTypingEffect() {
    const heroSubtitle = document.querySelector('.hero-subtitle');
    if (!heroSubtitle) return;
    
    const originalText = heroSubtitle.textContent;
    const typingTexts = [
        'Data Scientist & ML Engineer',
        'Machine Learning Specialist',
        'Analytics Expert',
        'Python Developer'
    ];
    
    let textIndex = 0;
    let charIndex = 0;
    let isDeleting = false;
    
    function typeText() {
        const currentText = typingTexts[textIndex];
        
        if (isDeleting) {
            heroSubtitle.textContent = currentText.substring(0, charIndex - 1);
            charIndex--;
        } else {
            heroSubtitle.textContent = currentText.substring(0, charIndex + 1);
            charIndex++;
        }
        
        let typeSpeed = isDeleting ? 50 : 100;
        
        if (!isDeleting && charIndex === currentText.length) {
            typeSpeed = 2000; // Pause at end
            isDeleting = true;
        } else if (isDeleting && charIndex === 0) {
            isDeleting = false;
            textIndex = (textIndex + 1) % typingTexts.length;
            typeSpeed = 500; // Pause before next word
        }
        
        setTimeout(typeText, typeSpeed);
    }
    
    // Start typing effect after a delay
    setTimeout(typeText, 1000);
}

// Contact form functionality
function initContactForm() {
    const contactForm = document.querySelector('#contact form');
    
    if (contactForm) {
        contactForm.addEventListener('submit', function(e) {
            const submitBtn = this.querySelector('button[type="submit"]');
            const originalText = submitBtn.textContent;
            
            // Show loading state
            submitBtn.disabled = true;
            submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Sending...';
            
            // Re-enable button after form submission
            setTimeout(() => {
                submitBtn.disabled = false;
                submitBtn.textContent = originalText;
            }, 2000);
        });
        
        // Form validation
        const inputs = contactForm.querySelectorAll('input, textarea');
        inputs.forEach(input => {
            input.addEventListener('blur', validateField);
            input.addEventListener('input', clearValidation);
        });
    }
}

function validateField(e) {
    const field = e.target;
    const value = field.value.trim();
    
    // Remove existing validation classes
    field.classList.remove('is-valid', 'is-invalid');
    
    // Validate based on field type
    let isValid = true;
    
    if (field.hasAttribute('required') && !value) {
        isValid = false;
    } else if (field.type === 'email' && value) {
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        isValid = emailRegex.test(value);
    }
    
    // Add validation class
    field.classList.add(isValid ? 'is-valid' : 'is-invalid');
}

function clearValidation(e) {
    const field = e.target;
    field.classList.remove('is-valid', 'is-invalid');
}

// Utility functions
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

function throttle(func, limit) {
    let inThrottle;
    return function() {
        const args = arguments;
        const context = this;
        if (!inThrottle) {
            func.apply(context, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    };
}

// Performance optimizations
window.addEventListener('scroll', throttle(function() {
    // Throttled scroll events
    updateActiveNavLink();
}, 100));

// Error handling
window.addEventListener('error', function(e) {
    console.error('JavaScript error:', e.error);
});

// Service worker registration (for PWA features)
if ('serviceWorker' in navigator) {
    window.addEventListener('load', function() {
        navigator.serviceWorker.register('/sw.js')
            .then(function(registration) {
                console.log('ServiceWorker registration successful');
            })
            .catch(function(error) {
                console.log('ServiceWorker registration failed');
            });
    });
}

// Analytics tracking (placeholder for future implementation)
function trackEvent(category, action, label) {
    // Placeholder for analytics tracking
    console.log('Event tracked:', { category, action, label });
}

// Track button clicks
document.addEventListener('click', function(e) {
    if (e.target.matches('.btn-primary, .btn-outline-light')) {
        const buttonText = e.target.textContent.trim();
        trackEvent('Button', 'Click', buttonText);
    }
});

// Lazy loading for images
function initLazyLoading() {
    const images = document.querySelectorAll('img[data-src]');
    
    const imageObserver = new IntersectionObserver((entries, observer) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const img = entry.target;
                img.src = img.dataset.src;
                img.classList.remove('lazy');
                observer.unobserve(img);
            }
        });
    });
    
    images.forEach(img => imageObserver.observe(img));
}

// Initialize lazy loading
initLazyLoading();

// Dark mode toggle (future enhancement)
function initDarkMode() {
    const darkModeToggle = document.querySelector('#dark-mode-toggle');
    
    if (darkModeToggle) {
        darkModeToggle.addEventListener('click', function() {
            document.body.classList.toggle('dark-mode');
            localStorage.setItem('darkMode', document.body.classList.contains('dark-mode'));
        });
        
        // Load saved dark mode preference
        if (localStorage.getItem('darkMode') === 'true') {
            document.body.classList.add('dark-mode');
        }
    }
}

// Initialize components that might be added later
document.addEventListener('DOMContentLoaded', function() {
    initDarkMode();
});
