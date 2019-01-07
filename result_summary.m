withz_res_a = load('./evaluation_data/fig4_z/nn_results_a647.mat');
withz_res_c = load('./evaluation_data/fig4_z/nn_results_c568.mat');
color_res 	= load('./evaluation_data/fig4_color/nn_results_color.mat');

yconv_z_a = withz_res_a.y_conv; 
coordinate = withz_res_a.y_coor; 
yconv_z_c = withz_res_c.y_conv; 
yconv_color = color_res.yconv_t; 
coordinate_color = color_res.ycoor; 
label_color = color_res.ylbl;
filter_a = label_color(:,1) == 1;
filter_c = label_color(:,1) == 0;

yconv_color_res = softmax(yconv_color');
yconv_color_res = yconv_color_res(1,:) - yconv_color_res(2,:);

photon_threshold = 3000;
threshold = 0.8;

a_ind = yconv_color_res >= threshold & coordinate(:,3)' >= photon_threshold;
c_ind = yconv_color_res <= -1 * threshold & coordinate(:,3)' >= photon_threshold;

a_coordinate = [coordinate(a_ind,1:2) -(yconv_z_a(a_ind))];
c_coordinate = [coordinate(c_ind,1:2) -(yconv_z_c(c_ind))];
t_coordinate = [a_coordinate ; c_coordinate];

res.a_coordinate = a_coordinate;
res.c_coordinate = c_coordinate;
res.t_coordinate = t_coordinate;

save('./evaluation_data/nn_results.mat','-struct','res');
