#include "hls_design_meta.h"
const Port_Property HLS_Design_Meta::port_props[]={
	Port_Property("ap_clk", 1, hls_in, -1, "", "", 1),
	Port_Property("ap_rst", 1, hls_in, -1, "", "", 1),
	Port_Property("ap_start", 1, hls_in, -1, "", "", 1),
	Port_Property("ap_done", 1, hls_out, -1, "", "", 1),
	Port_Property("ap_idle", 1, hls_out, -1, "", "", 1),
	Port_Property("ap_ready", 1, hls_out, -1, "", "", 1),
	Port_Property("x_profile_ap_vld", 1, hls_in, 0, "ap_vld", "in_vld", 1),
	Port_Property("z_global_ap_vld", 1, hls_in, 1, "ap_vld", "in_vld", 1),
	Port_Property("y_profile_ap_vld", 1, hls_in, 2, "ap_vld", "in_vld", 1),
	Port_Property("y_local_ap_vld", 1, hls_in, 3, "ap_vld", "in_vld", 1),
	Port_Property("x_profile", 336, hls_in, 0, "ap_vld", "in_data", 1),
	Port_Property("z_global", 16, hls_in, 1, "ap_vld", "in_data", 1),
	Port_Property("y_profile", 208, hls_in, 2, "ap_vld", "in_data", 1),
	Port_Property("y_local", 16, hls_in, 3, "ap_vld", "in_data", 1),
	Port_Property("layer25_out", 8, hls_out, 4, "ap_vld", "out_data", 1),
	Port_Property("layer25_out_ap_vld", 1, hls_out, 4, "ap_vld", "out_vld", 1),
};
const char* HLS_Design_Meta::dut_name = "myproject";
