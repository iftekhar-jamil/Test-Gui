let {PythonShell} = require('python-shell')
var path = require("path")


function get_weather() {

  var time = document.getElementById("time").value
  var date = document.getElementById("date").value
  // name = city
  var options = {
    scriptPath : path.join(__dirname, '/../engine/'),
    args : [time,date]
  }

  let pyshell = new PythonShell('weather_engine.py', options);


  pyshell.on('message', function(message) {
     console.log(message);
     var elem = document.createElement('img');
     elem.setAttribute("src", "data:image/jpeg;base64, "+message);
     elem.setAttribute("height", "400px");
     elem.setAttribute("width", "800px");
    // // newImage.src = message;
     document.getElementById("New").appendChild(elem);
    // // document.getElementById("New").innerHTML = newImage.outerHTML;
  })
  document.getElementById("time").value = "";
  document.getElementById("date").value = "";
  
}
