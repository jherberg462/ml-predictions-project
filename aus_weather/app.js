var predictions_url = 'http://localhost:5000/prediction/'
//url for endpoint for our predictions
//todo -- find documentation regarding how a flask endpoint will work on a heroku app

var directions = ['E', 'ENE', 'ESE', 'N', 'NE', 'NNE', 'NNW', 'NW', 'S',
'SE', 'SSE', 'SSW', 'SW', 'W', 'WNW', 'WSW'  ]
//list of possible wind directions (array -- js)

d3.select("#wind_gust_dir")
.selectAll("option")
.data(directions)
.enter()
.append('option')
.text(function(d){return d})
.attr("value", function(d){return d});
//have list of possible directions input into wind_gust_dir id 

d3.select("#wind_direction_9")
.selectAll("option")
.data(directions)
.enter()
.append('option')
.text(function(d){return d})
.attr("value", function(d){return d});

d3.select("#wind_direction_3")
.selectAll("option")
.data(directions)
.enter()
.append('option')
.text(function(d){return d})
.attr("value", function(d){return d});