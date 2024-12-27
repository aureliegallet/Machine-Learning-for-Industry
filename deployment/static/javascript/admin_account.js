const logout = () => {
    sessionStorage.removeItem('isAdmin');
    window.location.href = '/';
}

const validateLogin = event => {
    event.preventDefault();
    const username = document.getElementById('username').value;
    const password = document.getElementById('userpassword').value;

    // Note: in case of distribution to users, integrate authentication service for more security
    if (username === 'admin' && password === 'admin') {
        sessionStorage.setItem('isAdmin', 'true');

        // Display success message
        document.getElementById('message').innerHTML = '<div class="alert alert-success">Login successful! Redirecting...</div>';

        // Redirect the user after two seconds
        setTimeout(() => window.location.href = '/admin', 2000)
    } else {
        // Display invalid input message
        document.getElementById('message').innerHTML = '<div class="alert alert-danger">Invalid input.</div>';
    }
}