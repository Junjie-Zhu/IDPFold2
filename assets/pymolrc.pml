set cartoon_loop_radius, 0.33
set cartoon_oval_width, 0.33
set cartoon_rect_width, 0.33

set ray_shadows, 0
set ambient, 0.6
set direct, 0.5

set specular, 0.4
set shininess, 50

set ray_trace_mode, 1 
set ray_trace_gain, 0.2

bg_color white
set two_sided_lighting, on
hide spheres
zoom

set_color _blue, [134, 126, 174]
set_color _red, [252, 109, 73]

#set_color _blue, [59, 46, 126]
#set_color _red, [255, 51, 0]
alter_state 1, all, b = (x**2 + y**2 + z**2)**0.5
spectrum b, _blue _red