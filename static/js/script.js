/* ===============================
   LOGIN ROUTING
   (Mock routing removed. Handled by backend.)
   =============================== */

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
