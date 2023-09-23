const msgLabel = document.getElementById("msg-label")
const evalBtn = document.getElementById("Evaluate")
const modalResults = document.getElementById("modal-results")
const messageInput = document.getElementById("msg-input")
const settingsBtn = document.getElementById("settings-btn")
const config = document.getElementById("config")
const closeConfig = document.getElementById("close-config")
const closeResults = document.getElementById("close-results")
const checkBoxWord2Vec = document.getElementById("checkbox-word2vec")
const hiddenWord2Vec = document.querySelectorAll(".hidden-word2vec")
const naiveBayesBtn = document.getElementById("btn-naive-bayes")
const oneHotBtn = document.getElementById("btn-one-hot")
const skipGramBtn = document.getElementById("btn-skip-gram")
const word2vecBtn = document.getElementById("btn-word2vec")
const settingInput = document.querySelectorAll("input")
const closePlot = document.getElementById("close-plot")
const loadingScreen = document.getElementById("loading-screen")
loadingScreen.style.display = "none"
function createObject(object, variableName){
    globalThis[variableName] = object
}
naiveBayesBtn.addEventListener("click", () => {
    let naiveBayesTrain = pyscript.interpreter.globals.get('naiveBayesTrain')
    let trainingData = pyscript.interpreter.globals.get('trainingData')
    let spamData = pyscript.interpreter.globals.get('spamData')
    console.log(parseFloat(settingInput[1].value))
    setTimeout(function(){
        naiveBayesTrain(trainingData,spamData,parseInt(settingInput[2].value),parseFloat(settingInput[1].value))
    },10)
})
oneHotBtn.addEventListener("click", () => {
    loadingScreen.style.display = "block"
    let oneHotTrain = pyscript.interpreter.globals.get('oneHotTrain')
    let trainingData = pyscript.interpreter.globals.get('trainingData')
    let valData = pyscript.interpreter.globals.get('valData')
    let valSpam = pyscript.interpreter.globals.get('valSpam')
    let spamData = pyscript.interpreter.globals.get('spamData')
    setTimeout(function(){
        oneHotTrain(trainingData,spamData,valData,valSpam,parseInt(settingInput[3].value),parseInt(settingInput[4].value),parseFloat(settingInput[5].value),true)
        loadingScreen.style.display = "none"
        closePlot.style.display = "block"
        const plot = document.getElementById("plot")
        closePlot.addEventListener("click",() => {
            plot.innerHTML = ''
            closePlot.style.display = "none"
        })
        console.log("done")
    },10)
})
word2vecBtn.addEventListener("click", () => {
    loadingScreen.style.display = "block"
    let valData = pyscript.interpreter.globals.get('valData')
    let valSpam = pyscript.interpreter.globals.get('valSpam')
    let word2vecTrain = pyscript.interpreter.globals.get('word2vecTrain')
    let trainingData = pyscript.interpreter.globals.get('trainingData')
    let spamData = pyscript.interpreter.globals.get('spamData')
    setTimeout(function (){
        if (checkBoxWord2Vec.checked){
            word2vecTrain(trainingData,spamData,valData,valSpam,parseInt(settingInput[6].value),parseFloat(settingInput[7].value),true,2,parseInt(settingInput[9].value))
        }
        else{
            word2vecTrain(trainingData,spamData,valData,valSpam,parseInt(settingInput[6].value),parseFloat(settingInput[7].value),true,1)
        }
        loadingScreen.style.display = "none"
        closePlot.style.display = "block"
        const plot = document.getElementById("plot")
        closePlot.addEventListener("click",() => {
            plot.innerHTML = ''
            closePlot.style.display = "none"
        })
        console.log("done")
    },10)
})
skipGramBtn.addEventListener("click", () => {
    loadingScreen.style.display = "block"
    let valData = pyscript.interpreter.globals.get('valData')
    let valSpam = pyscript.interpreter.globals.get('valSpam')
    let skipGramTrain = pyscript.interpreter.globals.get('skipGramTrain')
    let trainingData = pyscript.interpreter.globals.get('trainingData')
    let spamData = pyscript.interpreter.globals.get('spamData')
    setTimeout(function (){
        skipGramTrain(trainingData,spamData,valData,valSpam,parseInt(settingInput[10].value),parseFloat(settingInput[11].value),parseInt(settingInput[12].value),parseInt(settingInput[13].value),parseFloat(settingInput[14].value))
        loadingScreen.style.display = "none"
        closePlot.style.display = "block"
        const plot = document.getElementById("plot")
        closePlot.addEventListener("click",() => {
            plot.innerHTML = ''
            closePlot.style.display = "none"
        })
        console.log("done")
    })
})
evalBtn.addEventListener("click", () => {
    modalResults.style.display = "block"
})
settingsBtn.addEventListener("click", () =>{
    config.style.display = "inline"
})
closeResults.addEventListener("click", () => {
    modalResults.style.display = "none"
    deleteChart()
})
closeConfig.addEventListener("click", () => {
    config.style.display = "none"
})
checkBoxWord2Vec.addEventListener("click", () => {
    if (checkBoxWord2Vec.checked){
        for (let i = 0; i<hiddenWord2Vec.length; i++){
            hiddenWord2Vec[i].style.display = "inline"
        }
        settingInput[7].value = 0.00002
    }
    else{
        for (let i = 0; i<hiddenWord2Vec.length; i++){
            hiddenWord2Vec[i].style.display = "none"
        }
        settingInput[7].value = 0.001
    }
})
function showX(){
    let message = messageInput.value
    loadMessage = pyscript.interpreter.globals.get('loadMessage')
    let oneHotModel = pyscript.interpreter.globals.get('one_hot_model')
    let wordProbHam = pyscript.interpreter.globals.get('wordProbHam') 
    let wordProbSpam = pyscript.interpreter.globals.get('wordProbSpam')
    let priorSpam = pyscript.interpreter.globals.get('priorSpam')
    let word2vecModel = pyscript.interpreter.globals.get('word2vec_model')
    let skipGramModel = pyscript.interpreter.globals.get('skip_gram_model')
    message = loadMessage(message)
    let naiveBayesFun = pyscript.interpreter.globals.get('analyseMsg')
    let oneHotFun = pyscript.interpreter.globals.get('useOneHot')
    let word2vecFun = pyscript.interpreter.globals.get('useWord2Vec')
    let skipGramTrained = pyscript.interpreter.globals.get('skipGramTrained')
    let useSkipGram = pyscript.interpreter.globals.get('useSkipGram')
    if (skipGramTrained === 1){
        modelResults = [naiveBayesFun(wordProbHam,wordProbSpam,message,priorSpam),oneHotFun(oneHotModel,message),word2vecFun(word2vecModel,message)]
    }
    else{
        modelResults = [naiveBayesFun(wordProbHam,wordProbSpam,message,priorSpam),oneHotFun(oneHotModel,message),word2vecFun(word2vecModel,message),useSkipGram(skipGramModel,message)]
    }
    createChart(modelResults)
}
function createChart(modelResults){
    console.log(modelResults)
    const margin = {top: 20, right: 30, bottom: 40, left: 90},
    width = 1000 - margin.left - margin.right,
    height = 400 - margin.top - margin.bottom;
    let titles = ["Naive Bayes", "One Hot encoding", "word2vec", "Skip Gram"]
    if (modelResults.length === 3){
        titles.splice(-1)
    }
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
class Buttons {
    constructor (button, color, borderStyle, backgroundColor) {
        this.button = button;
        this.button.style.color = color;
        this.button.style.border = borderStyle;
        this.button.style.backgroundColor = backgroundColor;
    }
    colorSwitch (newColor, newBorderStyle, newBackgroundColor) {
        this.button.style.color = newColor;
        this.button.style.border = newBorderStyle;
        this.button.style.backgroundColor = newBackgroundColor;
    }
}
const btns = document.querySelectorAll('button')
console.log(btns)
const buttonList = [];
let count = 1;
let count2 = 1;

for (let i = 0; i < btns.length; i++) {
    buttonList.push(new Buttons(btns[i], 'black', '1px solid black', 'aquamarine'));
    const element = buttonList[i];
    buttonList[i].button.addEventListener('mouseout', function () { btnHover1(element); });
    buttonList[i].button.addEventListener('mouseover', function () { btnHover2(element); });
}

console.log(buttonList)

function btnHover1 (button) {
    if (count % 2 === 1) {
        button.colorSwitch('black', '1px solid black', 'aquamarine');
    } else {
        button.colorSwitch('aquamarine', '1px solid aquamarine', 'black');
    };
    button.button.style.borderRadius = '20px';
}
function btnHover2 (element) {
    if (count % 2 === 0) {
        element.colorSwitch('black', '1px solid black', 'aquamarine');
    } else {
        element.colorSwitch('aquamarine', '1px solid aquamarine', 'black');
    }
    element.button.style.borderRadius = '0px';
}