
document.addEventListener("DOMContentLoaded",()=> {
    console.log("hi")
})
msgLabel = document.getElementById("msg-label")
evalBtn = document.getElementById("Evaluate")
modalResults = document.getElementById("modal-results")
evalBtn.addEventListener("click", () => {
    modalResults.style.display = "block"
    console.log("hi")
})
closeResults = document.getElementById("close-results")
closeResults.addEventListener("click", () => {
    modalResults.style.display = "none"
    console.log("hi")
})
