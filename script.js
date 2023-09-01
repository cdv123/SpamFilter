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
    deleteChart()
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
    let wordProbHam = pyscript.interpreter.globals.get('wordProbHam') 
    let word2vecWeights = pyscript.interpreter.globals.get('word2vecWeight')
    let word2vecBias = pyscript.interpreter.globals.get('word2vecBias')
    let wordProbSpam = pyscript.interpreter.globals.get('wordProbSpam')
    naivesBayesFun = pyscript.interpreter.globals.get('analyseMsg')
    message = loadMessage(message)
    oneHotFun = pyscript.interpreter.globals.get('useOneHot')
    word2vecFun =pyscript.interpreter.globals.get('useWord2Vec')
    modelResults = [naivesBayesFun(wordProbHam,wordProbSpam,message),oneHotFun(oneHotWeights,oneHotBias,message),word2vecFun(word2vecWeights,word2vecBias,message)]
    createChart(modelResults)
}
function createChart(modelResults){
    console.log(modelResults)
    const margin = {top: 20, right: 30, bottom: 40, left: 90},
    width = 1000 - margin.left - margin.right,
    height = 400 - margin.top - margin.bottom;
    const titles = ["Naive Bayes", "One Hot encoding", "word2vec"]
    let data = []
    for(let i = 0; i<titles.length;i++){
        data.push({title: titles[i], results: modelResults[i]})
    }
    // append the svg object to the body of the page
    const svg = d3.select("#my_dataviz")
    .append("svg")
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom)
    .append("g")
        .attr("transform", `translate(${margin.left}, ${margin.top})`);

    // Add X axis
    const x = d3.scaleLinear()
        .domain([0, 1])
        .range([ 0, width]);
    svg.append("g")
        .attr("transform", `translate(0, ${height})`)
        .call(d3.axisBottom(x))
        .selectAll("text")
        .attr("transform", "translate(-10,0)rotate(-45)")
        .style("text-anchor", "end");

    // Y axis
    const y = d3.scaleBand()
        .range([ 0, height ])   
        .domain(data.map(d => d.title))
        .padding(.5);
    svg.append("g")
        .call(d3.axisLeft(y))

    //Bars
    svg.selectAll("myRect")
        .data(data)
        .join("rect")
        .attr("x", x(0) )
        .attr("y", d => y(d.title))
        .attr("width", d => x(d.results))
        .attr("height", y.bandwidth())
        .attr("fill", "#69b3a2")
    } 
function deleteChart(){
    d3.select('svg').remove();
}
