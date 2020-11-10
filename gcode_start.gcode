; gcode_start version 1
; October 23, 2020
; This starting code purges the nozzle by printing around the perimeter of the bed once.
M75; start GLCD timer
G21; set units to Millimetres
G90; absolute positioning
M107; disable fans
M82; set extruder to absolute mode
G28 X0 Y0; home X and Y
G28 Z0; home Z
M203 X192 Y208 Z3; set limits on travel speed
; JOB PARAMETERS
; NOZZLE_DIAMETER_MM: 0.5
; LAYER_HEIGHT_MM: 0.5
M117 Heating...; LCD message: 
M190 S48.0; Bed temperature
M140 S60; Bed temperature
M109 R205.0; Nozzle temperature
G0 X0 Y0 Z15 F1500; 
G92 E0; set extruder position to 0
G1 F200 E0; prime the nozzle with filament
G92 E0; re-set extruder position to 0
M117 Purging nozzle...; LCD message: 
G0 X5.0 Y5.0 Z0.5 F1500; 
G1 X285.0 Y5.0 Z0.5 F1500 E10.972824639145754; 
G1 X285.0 Y270.0 Z0.5 F1500 E21.3578193869087; 
G1 X5.0 Y270.0 Z0.5 F1500 E32.33064402605446; 
G1 X5.0 Y10.0 Z0.5 F1500 E42.519695476689805; 
G92 E0; set extruder position to 0
; End of gcode_start