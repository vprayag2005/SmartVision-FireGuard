/* ===============================
   LOGIN ROUTING (kept inside IIFE)
   =============================== */
(function () {
  function qs(sel) {
    return document.querySelector(sel);
  }

  var loginForm = qs('[data-page="login"]');

  if (loginForm) {
    loginForm.addEventListener('submit', function (e) {
      e.preventDefault();

      var role = qs('#role');
      var roleValue = role ? role.value : 'User';

      if (roleValue === 'Admin') {
        window.location.href = 'admin.html';
      } else {
        window.location.href = 'monitoring.html';
      }
    });
  }
})();

/* ===============================
   MODAL FUNCTIONS (GLOBAL)
   =============================== */
function openModal(id) {
  var modal = document.getElementById(id);
  if (modal) {
    modal.style.display = 'flex';
  }
}

function closeModal(id) {
  var modal = document.getElementById(id);
  if (modal) {
    modal.style.display = 'none';
  }
}
