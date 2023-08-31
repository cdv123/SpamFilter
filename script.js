const msgLabel = document.getElementById("msg-label")
const evalBtn = document.getElementById("Evaluate")
const modalResults = document.getElementById("modal-results")
const messageInput = document.getElementById("msg-input")
evalBtn.addEventListener("click", () => {
    modalResults.style.display = "block"
})
let closeResults = document.getElementById("close-results")
closeResults.addEventListener("click", () => {
    modalResults.style.display = "none"
})
function trainModels(){
    trainingData = pyscript.interpreter.globals.get('trainingData')
    spamData = pyscript.interpreter.globals.get('spamData')
    oneHotTrain = pyscript.interpreter.globals.get('oneHotTrain')
    oneHotRes = oneHotTrain(trainingData,spamData,300)
    naivesBayesTrain = pyscript.interpreter.globals.get('naivesBayesTrain')
    wordProb = naivesBayesTrain(trainingData,spamData)
    let probHam = wordProb[0]
    let probSpam = wordProb[1]
    word2vecTrain = pyscript.interpreter.globals.get('word2vecTrain')
    word2vecRes = word2vecTrain(trainingData,spamData,300)
}
function showX(){
    let message = messageInput.value
    loadMessage = pyscript.interpreter.globals.get('loadMessage')
    let oneHotWeights = pyscript.interpreter.globals.get('weights')
    let oneHotBias = pyscript.interpreter.globals.get('bias')
    createChart(message)
}
function createChart(word){
    const colours = []
    const titles = ["Naive Bayes spam confidence", "One Hot Encoding spam confidence", "Custom Embedding spam confidence"]
    oneHotFun = pyscript.interpreter.globals.get('useNetwork')
    naivesBayesFun = pyscript.interpreter.globals.get('useModel')
    const margin = {top: 100, right:100, left:100, bottom:100}
    const height = 300
    const width = 300
    const chart = document.getElementById('results-chart')
    const svg = d3.select('#results-chart')
        .append('svg')
        .attr("width",300)
        .attr("height",300)
    const y = d3.scaleBand()
        .range([0,height])
        .domain(titles)
}
function deleteChart(){
    d3.select('svg').remove();

        
}
