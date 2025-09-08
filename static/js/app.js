/**
 * CropGuard AI - JavaScript Application
 * Handles client-side functionality for the crop disease detection system
 */

class CropGuardApp {
    constructor() {
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.checkSystemStatus();
        this.initializeTooltips();
        this.setupImagePreview();
    }

    setupEventListeners() {
        // Form validation
        document.querySelectorAll('form').forEach(form => {
            form.addEventListener('submit', this.handleFormSubmit.bind(this));
        });

        // File input validation
        document.querySelectorAll('input[type="file"]').forEach(input => {
            input.addEventListener('change', this.validateFileInput.bind(this));
        });

        // Navigation active state
        this.setActiveNavigation();
    }

    handleFormSubmit(event) {
        const form = event.target;
        const submitBtn = form.querySelector('button[type="submit"]');
        
        if (submitBtn) {
            // Add loading state
            this.setLoadingState(submitBtn, true);
            
            // Reset loading state after timeout (fallback)
            setTimeout(() => {
                this.setLoadingState(submitBtn, false);
            }, 30000);
        }
    }

    setLoadingState(button, isLoading) {
        if (isLoading) {
            button.originalText = button.innerHTML;
            button.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
            button.disabled = true;
        } else {
            if (button.originalText) {
                button.innerHTML = button.originalText;
            }
            button.disabled = false;
        }
    }

    validateFileInput(event) {
        const input = event.target;
        const file = input.files[0];
        
        if (!file) return;

        // Validate file type
        const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/bmp'];
        if (!allowedTypes.includes(file.type)) {
            this.showAlert('Please select a valid image file (JPG, PNG, GIF, BMP)', 'danger');
            input.value = '';
            return;
        }

        // Validate file size (16MB max)
        const maxSize = 16 * 1024 * 1024;
        if (file.size > maxSize) {
            this.showAlert('File size must be less than 16MB', 'danger');
            input.value = '';
            return;
        }

        // Update UI to show valid file
        input.classList.add('is-valid');
        input.classList.remove('is-invalid');
    }

    setupImagePreview() {
        const fileInputs = document.querySelectorAll('input[type="file"][accept*="image"]');
        
        fileInputs.forEach(input => {
            input.addEventListener('change', (event) => {
                const file = event.target.files[0];
                if (file) {
                    this.previewImage(file, input);
                }
            });
        });
    }

    previewImage(file, input) {
        const reader = new FileReader();
        reader.onload = (e) => {
            // Find or create preview container
            let previewContainer = input.parentNode.querySelector('.image-preview');
            if (!previewContainer) {
                previewContainer = document.createElement('div');
                previewContainer.className = 'image-preview mt-3';
                input.parentNode.appendChild(previewContainer);
            }

            previewContainer.innerHTML = `
                <div class="text-center">
                    <img src="${e.target.result}" 
                         class="img-fluid rounded border" 
                         style="max-height: 200px;" 
                         alt="Image preview">
                    <div class="mt-2">
                        <small class="text-muted">
                            File: ${file.name} (${this.formatFileSize(file.size)})
                        </small>
                    </div>
                </div>
            `;
        };
        reader.readAsDataURL(file);
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    setActiveNavigation() {
        const currentPath = window.location.pathname;
        const navLinks = document.querySelectorAll('.navbar-nav .nav-link');
        
        navLinks.forEach(link => {
            if (link.getAttribute('href') === currentPath) {
                link.classList.add('active');
            } else {
                link.classList.remove('active');
            }
        });
    }

    checkSystemStatus() {
        // Check if we're on the home page
        if (window.location.pathname === '/') {
            this.updateSystemStatus();
        }
    }

    updateSystemStatus() {
        const statusElement = document.getElementById('system-status');
        if (!statusElement) return;

        fetch('/api/disease_stats')
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    statusElement.innerHTML = `
                        <div class="alert alert-warning">
                            <i class="fas fa-exclamation-triangle"></i> 
                            System initializing... Please ensure model is trained.
                        </div>
                    `;
                } else {
                    statusElement.innerHTML = `
                        <div class="alert alert-success">
                            <i class="fas fa-check-circle"></i> System operational
                            <div class="mt-2">
                                <strong>Total Detections:</strong> ${data.total_detections || 0} | 
                                <strong>Recent Activity:</strong> ${data.recent_detections || 0}
                            </div>
                        </div>
                    `;
                }
            })
            .catch(error => {
                statusElement.innerHTML = `
                    <div class="alert alert-danger">
                        <i class="fas fa-times-circle"></i> 
                        Unable to connect to backend services
                    </div>
                `;
            });
    }

    initializeTooltips() {
        // Initialize Bootstrap tooltips if available
        if (typeof bootstrap !== 'undefined' && bootstrap.Tooltip) {
            const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
            tooltipTriggerList.map(function (tooltipTriggerEl) {
                return new bootstrap.Tooltip(tooltipTriggerEl);
            });
        }
    }

    showAlert(message, type = 'info', duration = 5000) {
        const alertContainer = document.querySelector('.container');
        if (!alertContainer) return;

        const alert = document.createElement('div');
        alert.className = `alert alert-${type} alert-dismissible fade show mt-3`;
        alert.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;

        alertContainer.insertBefore(alert, alertContainer.firstChild);

        // Auto-dismiss after duration
        setTimeout(() => {
            if (alert.parentNode) {
                alert.remove();
            }
        }, duration);
    }

    // Utility functions for camera/geolocation
    async getUserLocation() {
        return new Promise((resolve, reject) => {
            if (!navigator.geolocation) {
                reject(new Error('Geolocation is not supported'));
                return;
            }

            navigator.geolocation.getCurrentPosition(
                position => resolve({
                    latitude: position.coords.latitude,
                    longitude: position.coords.longitude,
                    accuracy: position.coords.accuracy
                }),
                error => reject(error),
                {
                    enableHighAccuracy: true,
                    timeout: 10000,
                    maximumAge: 300000 // 5 minutes
                }
            );
        });
    }

    formatConfidence(confidence) {
        return `${(confidence * 100).toFixed(1)}%`;
    }

    formatTimestamp(timestamp) {
        return new Date(timestamp).toLocaleString();
    }

    // Analytics and tracking
    trackEvent(eventName, properties = {}) {
        // Basic event tracking (can be extended with analytics services)
        console.log('Event:', eventName, properties);
        
        // Example: Send to analytics service
        // analytics.track(eventName, properties);
    }

    // Performance monitoring
    measurePerformance(name, fn) {
        const start = performance.now();
        const result = fn();
        const end = performance.now();
        
        console.log(`${name} took ${(end - start).toFixed(2)} milliseconds`);
        this.trackEvent('performance', {
            operation: name,
            duration: end - start
        });
        
        return result;
    }

    // Error handling
    handleError(error, context = '') {
        console.error('Application error:', error, context);
        
        // Track error
        this.trackEvent('error', {
            message: error.message,
            context: context,
            stack: error.stack
        });
        
        // Show user-friendly message
        this.showAlert(
            'An error occurred. Please try again or contact support if the problem persists.',
            'danger'
        );
    }
}

// Geolocation utilities
class GeolocationHelper {
    static async getCurrentPosition(options = {}) {
        const defaultOptions = {
            enableHighAccuracy: true,
            timeout: 10000,
            maximumAge: 300000
        };
        
        return new Promise((resolve, reject) => {
            if (!navigator.geolocation) {
                reject(new Error('Geolocation is not supported'));
                return;
            }
            
            navigator.geolocation.getCurrentPosition(
                resolve,
                reject,
                { ...defaultOptions, ...options }
            );
        });
    }
    
    static formatCoordinates(lat, lng, precision = 6) {
        return `${lat.toFixed(precision)}, ${lng.toFixed(precision)}`;
    }
    
    static calculateDistance(lat1, lng1, lat2, lng2) {
        const R = 6371; // Earth's radius in kilometers
        const dLat = this.toRadians(lat2 - lat1);
        const dLng = this.toRadians(lng2 - lng1);
        
        const a = Math.sin(dLat / 2) * Math.sin(dLat / 2) +
                Math.cos(this.toRadians(lat1)) * Math.cos(this.toRadians(lat2)) *
                Math.sin(dLng / 2) * Math.sin(dLng / 2);
        
        const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
        return R * c;
    }
    
    static toRadians(degrees) {
        return degrees * (Math.PI / 180);
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.cropGuardApp = new CropGuardApp();
});

// Export for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { CropGuardApp, GeolocationHelper };
}