var table_clim = ee.FeatureCollection("users/sirelkir/team_challenge/total_only_climate");
var table_reg = ee.FeatureCollection("users/sirelkir/team_challenge/total_complete");




// Add present data to climate predicted ones
var tableJoinF = ee.Filter.equals({
  leftField: 'FIPS Code',
  rightField: 'FIPS Code'
})
var join = ee.Join.inner()

var cleanJoinTable = function(feature){
  return ee.Feature(feature.get('primary')).copyProperties(feature.get('secondary'),['2014']);
}
table_clim = join.apply(table_clim,table_reg,tableJoinF).map(cleanJoinTable)

print(ee.List(table_clim.sort('FIPS Code').aggregate_array('2030')).zip(table_reg.sort('FIPS Code').aggregate_array('2030')).zip(table_reg.sort('FIPS Code').aggregate_array('FIPS Code')))
print(ee.List(table_clim.sort('FIPS Code').aggregate_array('2014')).zip(table_reg.sort('FIPS Code').aggregate_array('2014')))


var year = '2030'
var tableStr = 'clim'
var table = table_clim
var min= 0.5
var max= 2
var diverging = ['#3b4cc0','#dddddd','#c97182','#b40426']

var borders = ee.FeatureCollection('USDOS/LSIB_SIMPLE/2017');

var joinFilter = ee.Filter.equals({
  leftField: 'country_co',
  rightField: 'FIPS Code'
})
var join = ee.Join.inner()

var cleanJoin = function(feature){
  return ee.Feature(feature.get('primary')).copyProperties(feature.get('secondary'),null,['id','FIPS Code']);
}
var joinOutput = join.apply(borders,table,joinFilter).map(cleanJoin)
print(joinOutput.limit(5))

// Convert to relative
function toRelative(feature) { 
  var column= '2030';
  return feature.set(column,ee.Number(feature.get(column)).divide(feature.get('2014')))
}
joinOutput = joinOutput
  .filterMetadata('2014','not_equals',0)
  .filterMetadata('2030','not_equals',0)
  .map(toRelative)



// pick countries with no data
var joinInv = ee.Join.inverted()
var noData = joinInv.apply(borders,table,joinFilter)


// Visualization
var energy = ee.Image().float().paint(joinOutput,year);

var energyVis = {
  min: min,
  max: max,
  palette: diverging
  
}


var backgroundImg = ee.Image.rgb(
  ee.Image(255).uint8(),
  ee.Image(255).uint8(),
  ee.Image(255).uint8()
  )
var energyImg = energy.visualize(energyVis)
var noDataImg = noData.draw('f0f0f0',3,1)

var compositeImg = ee.ImageCollection([backgroundImg,noDataImg,energyImg]).mosaic()

Map.addLayer(backgroundImg)
Map.addLayer(noDataImg)
Map.addLayer(energy,energyVis,'energy')
Map.addLayer(compositeImg,null,'composite')

var worldAoi = ee.Geometry.Rectangle([-179.0, -58.0, 179.0, 78.0],null,false);
var box = ee.Geometry.Rectangle([-150.0, -58.0, 150.0, 78.0],null,false);


var arg = {
  dimensions: 400,
  region: box,
  // framesPerSecond: 2,
  crs: 'EPSG:3857'
};
    
print(ui.Thumbnail(compositeImg, arg));
print(compositeImg.getThumbURL(arg));




Export.image.toDrive({
  image: compositeImg,
  description: "countries_total_rel_"+tableStr+"_"+year+"_1200px",
  folder: "GEE Exports",
  region: worldAoi,
  dimensions: 1200,
  crs: 'EPSG:3857'
})
