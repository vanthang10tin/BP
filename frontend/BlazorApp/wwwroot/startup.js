// Keep a failed circuit startup visible instead of leaving an empty page.
const startup = document.getElementById('qc-startup');
const message = document.getElementById('qc-startup-message');
const slowStart = setTimeout(() => {
    if (message) {
        message.textContent = 'Máy chủ đang khởi động hoặc kết nối chậm. Vui lòng chờ hoặc tải lại trang.';
    }
}, 15000);

Promise.resolve().then(() => {
    if (typeof Blazor !== 'undefined' && Blazor.start) {
        return Blazor.start();
    }
    return Promise.resolve();
}).then(() => {
    clearTimeout(slowStart);
    if (startup) {
        startup.remove();
    }
}).catch(error => {
    clearTimeout(slowStart);
    if (message) {
        message.textContent = 'Không thể kết nối giao diện với máy chủ. Hãy tải lại trang để thử lại.';
    }
    console.error('QC Studio startup failed:', error);
});
