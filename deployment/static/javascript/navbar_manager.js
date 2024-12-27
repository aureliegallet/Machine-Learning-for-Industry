const isAdminLoggedIn = sessionStorage.getItem('isAdmin');
const adminButton = document.getElementById('adminButton');
const accountManagement = document.getElementById('accountManagement');

if (isAdminLoggedIn === 'true') {
    adminButton.classList.remove('disabled');
    adminButton.classList.remove('text-muted');
    adminButton.href = '/admin'; 
    accountManagement.innerText = 'Logout';
    accountManagement.href = '#';
    accountManagement.onclick = function() {
        sessionStorage.removeItem('isAdmin');
        location.reload();
    };
}