const msgLabel = document.getElementById("msg-label")
const evalBtn = document.getElementById("Evaluate")
const modalResults = document.getElementById("modal-results")
const messageInput = document.getElementById("msg-input")
const settingsBtn = document.getElementById("settings-btn")
const config = document.getElementById("config")
const closeConfig = document.getElementById("close-config")
const closeResults = document.getElementById("close-results")
const checkBoxOneHot = document.getElementById("checkbox-one-hot")
const checkBoxWord2Vec = document.getElementById("checkbox-word2vec")
const hiddenOneHot = document.querySelectorAll(".hidden-one-hot")
const hiddenWord2Vec = document.querySelectorAll(".hidden-word2vec")
const naiveBayesBtn = document.getElementById("btn-naive-bayes")
const oneHotBtn = document.getElementById("btn-one-hot")
const word2vecBtn = document.getElementById("btn-word2vec")
const settingInput = document.querySelectorAll("input")
function createObject(object, variableName){
    //Bind a variable whose name is the string variableName
    // to the object called 'object'
    let execString = variableName + " = object"
    eval(execString)
}

naiveBayesBtn.addEventListener("click", () => {
    var trainingData = pyscript.interpreter.globals.get('trainingData')
    var spamData = pyscript.interpreter.globals.get('spamData')
    let naiveBayesTrain = pyscript.interpreter.globals.get('naiveBayesTrain')
    console.log("hello")
    naiveBayesTrain(trainingData,spamData,parseInt(settingInput[2].value))
    for (const result of naiveResults){
        console.log(result["call"])
    }
})
oneHotBtn.addEventListener("click", () => {
})
word2vecBtn.addEventListener("click", () => {
})
evalBtn.addEventListener("click", () => {
    modalResults.style.display = "block"
})
settingsBtn.addEventListener("click", () =>{
    config.style.display = "inline"
    for (const animal of animals_from_py){
        console.log(animal)
    }

})
closeResults.addEventListener("click", () => {
    modalResults.style.display = "none"
    deleteChart()
})
closeConfig.addEventListener("click", () => {
    config.style.display = "none"
})
checkBoxOneHot.addEventListener("click", () => {
    if (checkBoxOneHot.checked){
        for (let i = 0; i<hiddenOneHot.length; i++){
            hiddenOneHot[i].style.display = "inline"
        }
    }
    else{
        for (let i = 0; i<hiddenWord2Vec.length; i++){
            hiddenOneHot[i].style.display = "none"
        }
    }
})
checkBoxWord2Vec.addEventListener("click", () => {
    if (checkBoxWord2Vec.checked){
        for (let i = 0; i<hiddenWord2Vec.length; i++){
            hiddenWord2Vec[i].style.display = "inline"
        }
    }
    else{
        for (let i = 0; i<hiddenWord2Vec.length; i++){
            hiddenWord2Vec[i].style.display = "none"
        }
    }
})
function trainModels(){
    trainingData = pyscript.interpreter.globals.get('trainingData')
    spamData = pyscript.interpreter.globals.get('spamData')
    oneHotTrain = pyscript.interpreter.globals.get('oneHotTrain')
    oneHotRes = oneHotTrain(trainingData,spamData,300)
    naiveBayesTrain = pyscript.interpreter.globals.get('naiveBayesTrain')
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
    naiveBayesFun = pyscript.interpreter.globals.get('analyseMsg')
    message = loadMessage(message)
    oneHotFun = pyscript.interpreter.globals.get('useOneHot')
    word2vecFun =pyscript.interpreter.globals.get('useWord2Vec')
    modelResults = [naiveBayesFun(wordProbHam,wordProbSpam,message),oneHotFun(oneHotWeights,oneHotBias,message),word2vecFun(word2vecWeights,word2vecBias,message)]
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
function showConfig(){

}