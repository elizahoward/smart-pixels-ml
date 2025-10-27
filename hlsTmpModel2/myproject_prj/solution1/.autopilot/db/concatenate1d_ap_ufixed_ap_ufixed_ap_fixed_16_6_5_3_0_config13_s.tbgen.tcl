set moduleName concatenate1d_ap_ufixed_ap_ufixed_ap_fixed_16_6_5_3_0_config13_s
set isTopModule 0
set isCombinational 1
set isDatapathOnly 0
set isPipelined 0
set pipeline_type function
set FunctionProtocol ap_ctrl_hs
set isOneStateSeq 0
set ProfileFlag 0
set StallSigGenFlag 0
set isEnableWaveformDebug 1
set hasInterrupt 0
set DLRegFirstOffset 0
set DLRegItemOffset 0
set C_modelName {concatenate1d<ap_ufixed,ap_ufixed,ap_fixed<16,6,5,3,0>,config13>}
set C_modelType { int 1024 }
set ap_memory_interface_dict [dict create]
set C_modelArgList {
	{ data1_0_val int 8 regular  }
	{ data1_1_val int 8 regular  }
	{ data1_2_val int 8 regular  }
	{ data1_3_val int 8 regular  }
	{ data1_4_val int 8 regular  }
	{ data1_5_val int 8 regular  }
	{ data1_6_val int 8 regular  }
	{ data1_7_val int 8 regular  }
	{ data1_8_val int 8 regular  }
	{ data1_9_val int 8 regular  }
	{ data1_10_val int 8 regular  }
	{ data1_11_val int 8 regular  }
	{ data1_12_val int 8 regular  }
	{ data1_13_val int 8 regular  }
	{ data1_14_val int 8 regular  }
	{ data1_15_val int 8 regular  }
	{ data1_16_val int 8 regular  }
	{ data1_17_val int 8 regular  }
	{ data1_18_val int 8 regular  }
	{ data1_19_val int 8 regular  }
	{ data1_20_val int 8 regular  }
	{ data1_21_val int 8 regular  }
	{ data1_22_val int 8 regular  }
	{ data1_23_val int 8 regular  }
	{ data1_24_val int 8 regular  }
	{ data1_25_val int 8 regular  }
	{ data1_26_val int 8 regular  }
	{ data1_27_val int 8 regular  }
	{ data1_28_val int 8 regular  }
	{ data1_29_val int 8 regular  }
	{ data1_30_val int 8 regular  }
	{ data1_31_val int 8 regular  }
	{ data2_0_val int 8 regular  }
	{ data2_1_val int 8 regular  }
	{ data2_2_val int 8 regular  }
	{ data2_3_val int 8 regular  }
	{ data2_4_val int 8 regular  }
	{ data2_5_val int 8 regular  }
	{ data2_6_val int 8 regular  }
	{ data2_7_val int 8 regular  }
	{ data2_8_val int 8 regular  }
	{ data2_9_val int 8 regular  }
	{ data2_10_val int 8 regular  }
	{ data2_11_val int 8 regular  }
	{ data2_12_val int 8 regular  }
	{ data2_13_val int 8 regular  }
	{ data2_14_val int 8 regular  }
	{ data2_15_val int 8 regular  }
	{ data2_16_val int 8 regular  }
	{ data2_17_val int 8 regular  }
	{ data2_18_val int 8 regular  }
	{ data2_19_val int 8 regular  }
	{ data2_20_val int 8 regular  }
	{ data2_21_val int 8 regular  }
	{ data2_22_val int 8 regular  }
	{ data2_23_val int 8 regular  }
	{ data2_24_val int 8 regular  }
	{ data2_25_val int 8 regular  }
	{ data2_26_val int 8 regular  }
	{ data2_27_val int 8 regular  }
	{ data2_28_val int 8 regular  }
	{ data2_29_val int 8 regular  }
	{ data2_30_val int 8 regular  }
	{ data2_31_val int 8 regular  }
}
set hasAXIMCache 0
set hasAXIML2Cache 0
set AXIMCacheInstDict [dict create]
set C_modelArgMapList {[ 
	{ "Name" : "data1_0_val", "interface" : "wire", "bitwidth" : 8, "direction" : "READONLY"} , 
 	{ "Name" : "data1_1_val", "interface" : "wire", "bitwidth" : 8, "direction" : "READONLY"} , 
 	{ "Name" : "data1_2_val", "interface" : "wire", "bitwidth" : 8, "direction" : "READONLY"} , 
 	{ "Name" : "data1_3_val", "interface" : "wire", "bitwidth" : 8, "direction" : "READONLY"} , 
 	{ "Name" : "data1_4_val", "interface" : "wire", "bitwidth" : 8, "direction" : "READONLY"} , 
 	{ "Name" : "data1_5_val", "interface" : "wire", "bitwidth" : 8, "direction" : "READONLY"} , 
 	{ "Name" : "data1_6_val", "interface" : "wire", "bitwidth" : 8, "direction" : "READONLY"} , 
 	{ "Name" : "data1_7_val", "interface" : "wire", "bitwidth" : 8, "direction" : "READONLY"} , 
 	{ "Name" : "data1_8_val", "interface" : "wire", "bitwidth" : 8, "direction" : "READONLY"} , 
 	{ "Name" : "data1_9_val", "interface" : "wire", "bitwidth" : 8, "direction" : "READONLY"} , 
 	{ "Name" : "data1_10_val", "interface" : "wire", "bitwidth" : 8, "direction" : "READONLY"} , 
 	{ "Name" : "data1_11_val", "interface" : "wire", "bitwidth" : 8, "direction" : "READONLY"} , 
 	{ "Name" : "data1_12_val", "interface" : "wire", "bitwidth" : 8, "direction" : "READONLY"} , 
 	{ "Name" : "data1_13_val", "interface" : "wire", "bitwidth" : 8, "direction" : "READONLY"} , 
 	{ "Name" : "data1_14_val", "interface" : "wire", "bitwidth" : 8, "direction" : "READONLY"} , 
 	{ "Name" : "data1_15_val", "interface" : "wire", "bitwidth" : 8, "direction" : "READONLY"} , 
 	{ "Name" : "data1_16_val", "interface" : "wire", "bitwidth" : 8, "direction" : "READONLY"} , 
 	{ "Name" : "data1_17_val", "interface" : "wire", "bitwidth" : 8, "direction" : "READONLY"} , 
 	{ "Name" : "data1_18_val", "interface" : "wire", "bitwidth" : 8, "direction" : "READONLY"} , 
 	{ "Name" : "data1_19_val", "interface" : "wire", "bitwidth" : 8, "direction" : "READONLY"} , 
 	{ "Name" : "data1_20_val", "interface" : "wire", "bitwidth" : 8, "direction" : "READONLY"} , 
 	{ "Name" : "data1_21_val", "interface" : "wire", "bitwidth" : 8, "direction" : "READONLY"} , 
 	{ "Name" : "data1_22_val", "interface" : "wire", "bitwidth" : 8, "direction" : "READONLY"} , 
 	{ "Name" : "data1_23_val", "interface" : "wire", "bitwidth" : 8, "direction" : "READONLY"} , 
 	{ "Name" : "data1_24_val", "interface" : "wire", "bitwidth" : 8, "direction" : "READONLY"} , 
 	{ "Name" : "data1_25_val", "interface" : "wire", "bitwidth" : 8, "direction" : "READONLY"} , 
 	{ "Name" : "data1_26_val", "interface" : "wire", "bitwidth" : 8, "direction" : "READONLY"} , 
 	{ "Name" : "data1_27_val", "interface" : "wire", "bitwidth" : 8, "direction" : "READONLY"} , 
 	{ "Name" : "data1_28_val", "interface" : "wire", "bitwidth" : 8, "direction" : "READONLY"} , 
 	{ "Name" : "data1_29_val", "interface" : "wire", "bitwidth" : 8, "direction" : "READONLY"} , 
 	{ "Name" : "data1_30_val", "interface" : "wire", "bitwidth" : 8, "direction" : "READONLY"} , 
 	{ "Name" : "data1_31_val", "interface" : "wire", "bitwidth" : 8, "direction" : "READONLY"} , 
 	{ "Name" : "data2_0_val", "interface" : "wire", "bitwidth" : 8, "direction" : "READONLY"} , 
 	{ "Name" : "data2_1_val", "interface" : "wire", "bitwidth" : 8, "direction" : "READONLY"} , 
 	{ "Name" : "data2_2_val", "interface" : "wire", "bitwidth" : 8, "direction" : "READONLY"} , 
 	{ "Name" : "data2_3_val", "interface" : "wire", "bitwidth" : 8, "direction" : "READONLY"} , 
 	{ "Name" : "data2_4_val", "interface" : "wire", "bitwidth" : 8, "direction" : "READONLY"} , 
 	{ "Name" : "data2_5_val", "interface" : "wire", "bitwidth" : 8, "direction" : "READONLY"} , 
 	{ "Name" : "data2_6_val", "interface" : "wire", "bitwidth" : 8, "direction" : "READONLY"} , 
 	{ "Name" : "data2_7_val", "interface" : "wire", "bitwidth" : 8, "direction" : "READONLY"} , 
 	{ "Name" : "data2_8_val", "interface" : "wire", "bitwidth" : 8, "direction" : "READONLY"} , 
 	{ "Name" : "data2_9_val", "interface" : "wire", "bitwidth" : 8, "direction" : "READONLY"} , 
 	{ "Name" : "data2_10_val", "interface" : "wire", "bitwidth" : 8, "direction" : "READONLY"} , 
 	{ "Name" : "data2_11_val", "interface" : "wire", "bitwidth" : 8, "direction" : "READONLY"} , 
 	{ "Name" : "data2_12_val", "interface" : "wire", "bitwidth" : 8, "direction" : "READONLY"} , 
 	{ "Name" : "data2_13_val", "interface" : "wire", "bitwidth" : 8, "direction" : "READONLY"} , 
 	{ "Name" : "data2_14_val", "interface" : "wire", "bitwidth" : 8, "direction" : "READONLY"} , 
 	{ "Name" : "data2_15_val", "interface" : "wire", "bitwidth" : 8, "direction" : "READONLY"} , 
 	{ "Name" : "data2_16_val", "interface" : "wire", "bitwidth" : 8, "direction" : "READONLY"} , 
 	{ "Name" : "data2_17_val", "interface" : "wire", "bitwidth" : 8, "direction" : "READONLY"} , 
 	{ "Name" : "data2_18_val", "interface" : "wire", "bitwidth" : 8, "direction" : "READONLY"} , 
 	{ "Name" : "data2_19_val", "interface" : "wire", "bitwidth" : 8, "direction" : "READONLY"} , 
 	{ "Name" : "data2_20_val", "interface" : "wire", "bitwidth" : 8, "direction" : "READONLY"} , 
 	{ "Name" : "data2_21_val", "interface" : "wire", "bitwidth" : 8, "direction" : "READONLY"} , 
 	{ "Name" : "data2_22_val", "interface" : "wire", "bitwidth" : 8, "direction" : "READONLY"} , 
 	{ "Name" : "data2_23_val", "interface" : "wire", "bitwidth" : 8, "direction" : "READONLY"} , 
 	{ "Name" : "data2_24_val", "interface" : "wire", "bitwidth" : 8, "direction" : "READONLY"} , 
 	{ "Name" : "data2_25_val", "interface" : "wire", "bitwidth" : 8, "direction" : "READONLY"} , 
 	{ "Name" : "data2_26_val", "interface" : "wire", "bitwidth" : 8, "direction" : "READONLY"} , 
 	{ "Name" : "data2_27_val", "interface" : "wire", "bitwidth" : 8, "direction" : "READONLY"} , 
 	{ "Name" : "data2_28_val", "interface" : "wire", "bitwidth" : 8, "direction" : "READONLY"} , 
 	{ "Name" : "data2_29_val", "interface" : "wire", "bitwidth" : 8, "direction" : "READONLY"} , 
 	{ "Name" : "data2_30_val", "interface" : "wire", "bitwidth" : 8, "direction" : "READONLY"} , 
 	{ "Name" : "data2_31_val", "interface" : "wire", "bitwidth" : 8, "direction" : "READONLY"} , 
 	{ "Name" : "ap_return", "interface" : "wire", "bitwidth" : 1024} ]}
# RTL Port declarations: 
set portNum 130
set portList { 
	{ ap_ready sc_out sc_logic 1 ready -1 } 
	{ data1_0_val sc_in sc_lv 8 signal 0 } 
	{ data1_1_val sc_in sc_lv 8 signal 1 } 
	{ data1_2_val sc_in sc_lv 8 signal 2 } 
	{ data1_3_val sc_in sc_lv 8 signal 3 } 
	{ data1_4_val sc_in sc_lv 8 signal 4 } 
	{ data1_5_val sc_in sc_lv 8 signal 5 } 
	{ data1_6_val sc_in sc_lv 8 signal 6 } 
	{ data1_7_val sc_in sc_lv 8 signal 7 } 
	{ data1_8_val sc_in sc_lv 8 signal 8 } 
	{ data1_9_val sc_in sc_lv 8 signal 9 } 
	{ data1_10_val sc_in sc_lv 8 signal 10 } 
	{ data1_11_val sc_in sc_lv 8 signal 11 } 
	{ data1_12_val sc_in sc_lv 8 signal 12 } 
	{ data1_13_val sc_in sc_lv 8 signal 13 } 
	{ data1_14_val sc_in sc_lv 8 signal 14 } 
	{ data1_15_val sc_in sc_lv 8 signal 15 } 
	{ data1_16_val sc_in sc_lv 8 signal 16 } 
	{ data1_17_val sc_in sc_lv 8 signal 17 } 
	{ data1_18_val sc_in sc_lv 8 signal 18 } 
	{ data1_19_val sc_in sc_lv 8 signal 19 } 
	{ data1_20_val sc_in sc_lv 8 signal 20 } 
	{ data1_21_val sc_in sc_lv 8 signal 21 } 
	{ data1_22_val sc_in sc_lv 8 signal 22 } 
	{ data1_23_val sc_in sc_lv 8 signal 23 } 
	{ data1_24_val sc_in sc_lv 8 signal 24 } 
	{ data1_25_val sc_in sc_lv 8 signal 25 } 
	{ data1_26_val sc_in sc_lv 8 signal 26 } 
	{ data1_27_val sc_in sc_lv 8 signal 27 } 
	{ data1_28_val sc_in sc_lv 8 signal 28 } 
	{ data1_29_val sc_in sc_lv 8 signal 29 } 
	{ data1_30_val sc_in sc_lv 8 signal 30 } 
	{ data1_31_val sc_in sc_lv 8 signal 31 } 
	{ data2_0_val sc_in sc_lv 8 signal 32 } 
	{ data2_1_val sc_in sc_lv 8 signal 33 } 
	{ data2_2_val sc_in sc_lv 8 signal 34 } 
	{ data2_3_val sc_in sc_lv 8 signal 35 } 
	{ data2_4_val sc_in sc_lv 8 signal 36 } 
	{ data2_5_val sc_in sc_lv 8 signal 37 } 
	{ data2_6_val sc_in sc_lv 8 signal 38 } 
	{ data2_7_val sc_in sc_lv 8 signal 39 } 
	{ data2_8_val sc_in sc_lv 8 signal 40 } 
	{ data2_9_val sc_in sc_lv 8 signal 41 } 
	{ data2_10_val sc_in sc_lv 8 signal 42 } 
	{ data2_11_val sc_in sc_lv 8 signal 43 } 
	{ data2_12_val sc_in sc_lv 8 signal 44 } 
	{ data2_13_val sc_in sc_lv 8 signal 45 } 
	{ data2_14_val sc_in sc_lv 8 signal 46 } 
	{ data2_15_val sc_in sc_lv 8 signal 47 } 
	{ data2_16_val sc_in sc_lv 8 signal 48 } 
	{ data2_17_val sc_in sc_lv 8 signal 49 } 
	{ data2_18_val sc_in sc_lv 8 signal 50 } 
	{ data2_19_val sc_in sc_lv 8 signal 51 } 
	{ data2_20_val sc_in sc_lv 8 signal 52 } 
	{ data2_21_val sc_in sc_lv 8 signal 53 } 
	{ data2_22_val sc_in sc_lv 8 signal 54 } 
	{ data2_23_val sc_in sc_lv 8 signal 55 } 
	{ data2_24_val sc_in sc_lv 8 signal 56 } 
	{ data2_25_val sc_in sc_lv 8 signal 57 } 
	{ data2_26_val sc_in sc_lv 8 signal 58 } 
	{ data2_27_val sc_in sc_lv 8 signal 59 } 
	{ data2_28_val sc_in sc_lv 8 signal 60 } 
	{ data2_29_val sc_in sc_lv 8 signal 61 } 
	{ data2_30_val sc_in sc_lv 8 signal 62 } 
	{ data2_31_val sc_in sc_lv 8 signal 63 } 
	{ ap_return_0 sc_out sc_lv 16 signal -1 } 
	{ ap_return_1 sc_out sc_lv 16 signal -1 } 
	{ ap_return_2 sc_out sc_lv 16 signal -1 } 
	{ ap_return_3 sc_out sc_lv 16 signal -1 } 
	{ ap_return_4 sc_out sc_lv 16 signal -1 } 
	{ ap_return_5 sc_out sc_lv 16 signal -1 } 
	{ ap_return_6 sc_out sc_lv 16 signal -1 } 
	{ ap_return_7 sc_out sc_lv 16 signal -1 } 
	{ ap_return_8 sc_out sc_lv 16 signal -1 } 
	{ ap_return_9 sc_out sc_lv 16 signal -1 } 
	{ ap_return_10 sc_out sc_lv 16 signal -1 } 
	{ ap_return_11 sc_out sc_lv 16 signal -1 } 
	{ ap_return_12 sc_out sc_lv 16 signal -1 } 
	{ ap_return_13 sc_out sc_lv 16 signal -1 } 
	{ ap_return_14 sc_out sc_lv 16 signal -1 } 
	{ ap_return_15 sc_out sc_lv 16 signal -1 } 
	{ ap_return_16 sc_out sc_lv 16 signal -1 } 
	{ ap_return_17 sc_out sc_lv 16 signal -1 } 
	{ ap_return_18 sc_out sc_lv 16 signal -1 } 
	{ ap_return_19 sc_out sc_lv 16 signal -1 } 
	{ ap_return_20 sc_out sc_lv 16 signal -1 } 
	{ ap_return_21 sc_out sc_lv 16 signal -1 } 
	{ ap_return_22 sc_out sc_lv 16 signal -1 } 
	{ ap_return_23 sc_out sc_lv 16 signal -1 } 
	{ ap_return_24 sc_out sc_lv 16 signal -1 } 
	{ ap_return_25 sc_out sc_lv 16 signal -1 } 
	{ ap_return_26 sc_out sc_lv 16 signal -1 } 
	{ ap_return_27 sc_out sc_lv 16 signal -1 } 
	{ ap_return_28 sc_out sc_lv 16 signal -1 } 
	{ ap_return_29 sc_out sc_lv 16 signal -1 } 
	{ ap_return_30 sc_out sc_lv 16 signal -1 } 
	{ ap_return_31 sc_out sc_lv 16 signal -1 } 
	{ ap_return_32 sc_out sc_lv 16 signal -1 } 
	{ ap_return_33 sc_out sc_lv 16 signal -1 } 
	{ ap_return_34 sc_out sc_lv 16 signal -1 } 
	{ ap_return_35 sc_out sc_lv 16 signal -1 } 
	{ ap_return_36 sc_out sc_lv 16 signal -1 } 
	{ ap_return_37 sc_out sc_lv 16 signal -1 } 
	{ ap_return_38 sc_out sc_lv 16 signal -1 } 
	{ ap_return_39 sc_out sc_lv 16 signal -1 } 
	{ ap_return_40 sc_out sc_lv 16 signal -1 } 
	{ ap_return_41 sc_out sc_lv 16 signal -1 } 
	{ ap_return_42 sc_out sc_lv 16 signal -1 } 
	{ ap_return_43 sc_out sc_lv 16 signal -1 } 
	{ ap_return_44 sc_out sc_lv 16 signal -1 } 
	{ ap_return_45 sc_out sc_lv 16 signal -1 } 
	{ ap_return_46 sc_out sc_lv 16 signal -1 } 
	{ ap_return_47 sc_out sc_lv 16 signal -1 } 
	{ ap_return_48 sc_out sc_lv 16 signal -1 } 
	{ ap_return_49 sc_out sc_lv 16 signal -1 } 
	{ ap_return_50 sc_out sc_lv 16 signal -1 } 
	{ ap_return_51 sc_out sc_lv 16 signal -1 } 
	{ ap_return_52 sc_out sc_lv 16 signal -1 } 
	{ ap_return_53 sc_out sc_lv 16 signal -1 } 
	{ ap_return_54 sc_out sc_lv 16 signal -1 } 
	{ ap_return_55 sc_out sc_lv 16 signal -1 } 
	{ ap_return_56 sc_out sc_lv 16 signal -1 } 
	{ ap_return_57 sc_out sc_lv 16 signal -1 } 
	{ ap_return_58 sc_out sc_lv 16 signal -1 } 
	{ ap_return_59 sc_out sc_lv 16 signal -1 } 
	{ ap_return_60 sc_out sc_lv 16 signal -1 } 
	{ ap_return_61 sc_out sc_lv 16 signal -1 } 
	{ ap_return_62 sc_out sc_lv 16 signal -1 } 
	{ ap_return_63 sc_out sc_lv 16 signal -1 } 
	{ ap_rst sc_in sc_logic 1 reset -1 active_high_sync } 
}
set NewPortList {[ 
	{ "name": "ap_ready", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "ready", "bundle":{"name": "ap_ready", "role": "default" }} , 
 	{ "name": "data1_0_val", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "data1_0_val", "role": "default" }} , 
 	{ "name": "data1_1_val", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "data1_1_val", "role": "default" }} , 
 	{ "name": "data1_2_val", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "data1_2_val", "role": "default" }} , 
 	{ "name": "data1_3_val", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "data1_3_val", "role": "default" }} , 
 	{ "name": "data1_4_val", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "data1_4_val", "role": "default" }} , 
 	{ "name": "data1_5_val", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "data1_5_val", "role": "default" }} , 
 	{ "name": "data1_6_val", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "data1_6_val", "role": "default" }} , 
 	{ "name": "data1_7_val", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "data1_7_val", "role": "default" }} , 
 	{ "name": "data1_8_val", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "data1_8_val", "role": "default" }} , 
 	{ "name": "data1_9_val", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "data1_9_val", "role": "default" }} , 
 	{ "name": "data1_10_val", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "data1_10_val", "role": "default" }} , 
 	{ "name": "data1_11_val", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "data1_11_val", "role": "default" }} , 
 	{ "name": "data1_12_val", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "data1_12_val", "role": "default" }} , 
 	{ "name": "data1_13_val", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "data1_13_val", "role": "default" }} , 
 	{ "name": "data1_14_val", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "data1_14_val", "role": "default" }} , 
 	{ "name": "data1_15_val", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "data1_15_val", "role": "default" }} , 
 	{ "name": "data1_16_val", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "data1_16_val", "role": "default" }} , 
 	{ "name": "data1_17_val", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "data1_17_val", "role": "default" }} , 
 	{ "name": "data1_18_val", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "data1_18_val", "role": "default" }} , 
 	{ "name": "data1_19_val", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "data1_19_val", "role": "default" }} , 
 	{ "name": "data1_20_val", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "data1_20_val", "role": "default" }} , 
 	{ "name": "data1_21_val", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "data1_21_val", "role": "default" }} , 
 	{ "name": "data1_22_val", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "data1_22_val", "role": "default" }} , 
 	{ "name": "data1_23_val", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "data1_23_val", "role": "default" }} , 
 	{ "name": "data1_24_val", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "data1_24_val", "role": "default" }} , 
 	{ "name": "data1_25_val", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "data1_25_val", "role": "default" }} , 
 	{ "name": "data1_26_val", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "data1_26_val", "role": "default" }} , 
 	{ "name": "data1_27_val", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "data1_27_val", "role": "default" }} , 
 	{ "name": "data1_28_val", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "data1_28_val", "role": "default" }} , 
 	{ "name": "data1_29_val", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "data1_29_val", "role": "default" }} , 
 	{ "name": "data1_30_val", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "data1_30_val", "role": "default" }} , 
 	{ "name": "data1_31_val", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "data1_31_val", "role": "default" }} , 
 	{ "name": "data2_0_val", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "data2_0_val", "role": "default" }} , 
 	{ "name": "data2_1_val", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "data2_1_val", "role": "default" }} , 
 	{ "name": "data2_2_val", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "data2_2_val", "role": "default" }} , 
 	{ "name": "data2_3_val", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "data2_3_val", "role": "default" }} , 
 	{ "name": "data2_4_val", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "data2_4_val", "role": "default" }} , 
 	{ "name": "data2_5_val", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "data2_5_val", "role": "default" }} , 
 	{ "name": "data2_6_val", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "data2_6_val", "role": "default" }} , 
 	{ "name": "data2_7_val", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "data2_7_val", "role": "default" }} , 
 	{ "name": "data2_8_val", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "data2_8_val", "role": "default" }} , 
 	{ "name": "data2_9_val", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "data2_9_val", "role": "default" }} , 
 	{ "name": "data2_10_val", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "data2_10_val", "role": "default" }} , 
 	{ "name": "data2_11_val", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "data2_11_val", "role": "default" }} , 
 	{ "name": "data2_12_val", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "data2_12_val", "role": "default" }} , 
 	{ "name": "data2_13_val", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "data2_13_val", "role": "default" }} , 
 	{ "name": "data2_14_val", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "data2_14_val", "role": "default" }} , 
 	{ "name": "data2_15_val", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "data2_15_val", "role": "default" }} , 
 	{ "name": "data2_16_val", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "data2_16_val", "role": "default" }} , 
 	{ "name": "data2_17_val", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "data2_17_val", "role": "default" }} , 
 	{ "name": "data2_18_val", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "data2_18_val", "role": "default" }} , 
 	{ "name": "data2_19_val", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "data2_19_val", "role": "default" }} , 
 	{ "name": "data2_20_val", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "data2_20_val", "role": "default" }} , 
 	{ "name": "data2_21_val", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "data2_21_val", "role": "default" }} , 
 	{ "name": "data2_22_val", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "data2_22_val", "role": "default" }} , 
 	{ "name": "data2_23_val", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "data2_23_val", "role": "default" }} , 
 	{ "name": "data2_24_val", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "data2_24_val", "role": "default" }} , 
 	{ "name": "data2_25_val", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "data2_25_val", "role": "default" }} , 
 	{ "name": "data2_26_val", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "data2_26_val", "role": "default" }} , 
 	{ "name": "data2_27_val", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "data2_27_val", "role": "default" }} , 
 	{ "name": "data2_28_val", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "data2_28_val", "role": "default" }} , 
 	{ "name": "data2_29_val", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "data2_29_val", "role": "default" }} , 
 	{ "name": "data2_30_val", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "data2_30_val", "role": "default" }} , 
 	{ "name": "data2_31_val", "direction": "in", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "data2_31_val", "role": "default" }} , 
 	{ "name": "ap_return_0", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "ap_return_0", "role": "default" }} , 
 	{ "name": "ap_return_1", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "ap_return_1", "role": "default" }} , 
 	{ "name": "ap_return_2", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "ap_return_2", "role": "default" }} , 
 	{ "name": "ap_return_3", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "ap_return_3", "role": "default" }} , 
 	{ "name": "ap_return_4", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "ap_return_4", "role": "default" }} , 
 	{ "name": "ap_return_5", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "ap_return_5", "role": "default" }} , 
 	{ "name": "ap_return_6", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "ap_return_6", "role": "default" }} , 
 	{ "name": "ap_return_7", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "ap_return_7", "role": "default" }} , 
 	{ "name": "ap_return_8", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "ap_return_8", "role": "default" }} , 
 	{ "name": "ap_return_9", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "ap_return_9", "role": "default" }} , 
 	{ "name": "ap_return_10", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "ap_return_10", "role": "default" }} , 
 	{ "name": "ap_return_11", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "ap_return_11", "role": "default" }} , 
 	{ "name": "ap_return_12", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "ap_return_12", "role": "default" }} , 
 	{ "name": "ap_return_13", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "ap_return_13", "role": "default" }} , 
 	{ "name": "ap_return_14", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "ap_return_14", "role": "default" }} , 
 	{ "name": "ap_return_15", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "ap_return_15", "role": "default" }} , 
 	{ "name": "ap_return_16", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "ap_return_16", "role": "default" }} , 
 	{ "name": "ap_return_17", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "ap_return_17", "role": "default" }} , 
 	{ "name": "ap_return_18", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "ap_return_18", "role": "default" }} , 
 	{ "name": "ap_return_19", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "ap_return_19", "role": "default" }} , 
 	{ "name": "ap_return_20", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "ap_return_20", "role": "default" }} , 
 	{ "name": "ap_return_21", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "ap_return_21", "role": "default" }} , 
 	{ "name": "ap_return_22", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "ap_return_22", "role": "default" }} , 
 	{ "name": "ap_return_23", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "ap_return_23", "role": "default" }} , 
 	{ "name": "ap_return_24", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "ap_return_24", "role": "default" }} , 
 	{ "name": "ap_return_25", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "ap_return_25", "role": "default" }} , 
 	{ "name": "ap_return_26", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "ap_return_26", "role": "default" }} , 
 	{ "name": "ap_return_27", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "ap_return_27", "role": "default" }} , 
 	{ "name": "ap_return_28", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "ap_return_28", "role": "default" }} , 
 	{ "name": "ap_return_29", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "ap_return_29", "role": "default" }} , 
 	{ "name": "ap_return_30", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "ap_return_30", "role": "default" }} , 
 	{ "name": "ap_return_31", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "ap_return_31", "role": "default" }} , 
 	{ "name": "ap_return_32", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "ap_return_32", "role": "default" }} , 
 	{ "name": "ap_return_33", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "ap_return_33", "role": "default" }} , 
 	{ "name": "ap_return_34", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "ap_return_34", "role": "default" }} , 
 	{ "name": "ap_return_35", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "ap_return_35", "role": "default" }} , 
 	{ "name": "ap_return_36", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "ap_return_36", "role": "default" }} , 
 	{ "name": "ap_return_37", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "ap_return_37", "role": "default" }} , 
 	{ "name": "ap_return_38", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "ap_return_38", "role": "default" }} , 
 	{ "name": "ap_return_39", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "ap_return_39", "role": "default" }} , 
 	{ "name": "ap_return_40", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "ap_return_40", "role": "default" }} , 
 	{ "name": "ap_return_41", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "ap_return_41", "role": "default" }} , 
 	{ "name": "ap_return_42", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "ap_return_42", "role": "default" }} , 
 	{ "name": "ap_return_43", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "ap_return_43", "role": "default" }} , 
 	{ "name": "ap_return_44", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "ap_return_44", "role": "default" }} , 
 	{ "name": "ap_return_45", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "ap_return_45", "role": "default" }} , 
 	{ "name": "ap_return_46", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "ap_return_46", "role": "default" }} , 
 	{ "name": "ap_return_47", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "ap_return_47", "role": "default" }} , 
 	{ "name": "ap_return_48", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "ap_return_48", "role": "default" }} , 
 	{ "name": "ap_return_49", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "ap_return_49", "role": "default" }} , 
 	{ "name": "ap_return_50", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "ap_return_50", "role": "default" }} , 
 	{ "name": "ap_return_51", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "ap_return_51", "role": "default" }} , 
 	{ "name": "ap_return_52", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "ap_return_52", "role": "default" }} , 
 	{ "name": "ap_return_53", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "ap_return_53", "role": "default" }} , 
 	{ "name": "ap_return_54", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "ap_return_54", "role": "default" }} , 
 	{ "name": "ap_return_55", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "ap_return_55", "role": "default" }} , 
 	{ "name": "ap_return_56", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "ap_return_56", "role": "default" }} , 
 	{ "name": "ap_return_57", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "ap_return_57", "role": "default" }} , 
 	{ "name": "ap_return_58", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "ap_return_58", "role": "default" }} , 
 	{ "name": "ap_return_59", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "ap_return_59", "role": "default" }} , 
 	{ "name": "ap_return_60", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "ap_return_60", "role": "default" }} , 
 	{ "name": "ap_return_61", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "ap_return_61", "role": "default" }} , 
 	{ "name": "ap_return_62", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "ap_return_62", "role": "default" }} , 
 	{ "name": "ap_return_63", "direction": "out", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "ap_return_63", "role": "default" }} , 
 	{ "name": "ap_rst", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "reset", "bundle":{"name": "ap_rst", "role": "default" }}  ]}

set RtlHierarchyInfo {[
	{"ID" : "0", "Level" : "0", "Path" : "`AUTOTB_DUT_INST", "Parent" : "",
		"CDFG" : "concatenate1d_ap_ufixed_ap_ufixed_ap_fixed_16_6_5_3_0_config13_s",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "0", "ap_start" : "0", "ap_ready" : "1", "ap_done" : "0", "ap_continue" : "0", "ap_idle" : "0", "real_start" : "0",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "1",
		"VariableLatency" : "0", "ExactLatency" : "0", "EstimateLatencyMin" : "0", "EstimateLatencyMax" : "0",
		"Combinational" : "1",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"IsBlackBox" : "0",
		"Port" : [
			{"Name" : "data1_0_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data1_1_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data1_2_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data1_3_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data1_4_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data1_5_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data1_6_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data1_7_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data1_8_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data1_9_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data1_10_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data1_11_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data1_12_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data1_13_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data1_14_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data1_15_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data1_16_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data1_17_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data1_18_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data1_19_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data1_20_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data1_21_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data1_22_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data1_23_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data1_24_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data1_25_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data1_26_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data1_27_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data1_28_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data1_29_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data1_30_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data1_31_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data2_0_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data2_1_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data2_2_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data2_3_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data2_4_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data2_5_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data2_6_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data2_7_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data2_8_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data2_9_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data2_10_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data2_11_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data2_12_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data2_13_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data2_14_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data2_15_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data2_16_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data2_17_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data2_18_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data2_19_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data2_20_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data2_21_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data2_22_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data2_23_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data2_24_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data2_25_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data2_26_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data2_27_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data2_28_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data2_29_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data2_30_val", "Type" : "None", "Direction" : "I"},
			{"Name" : "data2_31_val", "Type" : "None", "Direction" : "I"}]}]}


set ArgLastReadFirstWriteLatency {
	concatenate1d_ap_ufixed_ap_ufixed_ap_fixed_16_6_5_3_0_config13_s {
		data1_0_val {Type I LastRead 0 FirstWrite -1}
		data1_1_val {Type I LastRead 0 FirstWrite -1}
		data1_2_val {Type I LastRead 0 FirstWrite -1}
		data1_3_val {Type I LastRead 0 FirstWrite -1}
		data1_4_val {Type I LastRead 0 FirstWrite -1}
		data1_5_val {Type I LastRead 0 FirstWrite -1}
		data1_6_val {Type I LastRead 0 FirstWrite -1}
		data1_7_val {Type I LastRead 0 FirstWrite -1}
		data1_8_val {Type I LastRead 0 FirstWrite -1}
		data1_9_val {Type I LastRead 0 FirstWrite -1}
		data1_10_val {Type I LastRead 0 FirstWrite -1}
		data1_11_val {Type I LastRead 0 FirstWrite -1}
		data1_12_val {Type I LastRead 0 FirstWrite -1}
		data1_13_val {Type I LastRead 0 FirstWrite -1}
		data1_14_val {Type I LastRead 0 FirstWrite -1}
		data1_15_val {Type I LastRead 0 FirstWrite -1}
		data1_16_val {Type I LastRead 0 FirstWrite -1}
		data1_17_val {Type I LastRead 0 FirstWrite -1}
		data1_18_val {Type I LastRead 0 FirstWrite -1}
		data1_19_val {Type I LastRead 0 FirstWrite -1}
		data1_20_val {Type I LastRead 0 FirstWrite -1}
		data1_21_val {Type I LastRead 0 FirstWrite -1}
		data1_22_val {Type I LastRead 0 FirstWrite -1}
		data1_23_val {Type I LastRead 0 FirstWrite -1}
		data1_24_val {Type I LastRead 0 FirstWrite -1}
		data1_25_val {Type I LastRead 0 FirstWrite -1}
		data1_26_val {Type I LastRead 0 FirstWrite -1}
		data1_27_val {Type I LastRead 0 FirstWrite -1}
		data1_28_val {Type I LastRead 0 FirstWrite -1}
		data1_29_val {Type I LastRead 0 FirstWrite -1}
		data1_30_val {Type I LastRead 0 FirstWrite -1}
		data1_31_val {Type I LastRead 0 FirstWrite -1}
		data2_0_val {Type I LastRead 0 FirstWrite -1}
		data2_1_val {Type I LastRead 0 FirstWrite -1}
		data2_2_val {Type I LastRead 0 FirstWrite -1}
		data2_3_val {Type I LastRead 0 FirstWrite -1}
		data2_4_val {Type I LastRead 0 FirstWrite -1}
		data2_5_val {Type I LastRead 0 FirstWrite -1}
		data2_6_val {Type I LastRead 0 FirstWrite -1}
		data2_7_val {Type I LastRead 0 FirstWrite -1}
		data2_8_val {Type I LastRead 0 FirstWrite -1}
		data2_9_val {Type I LastRead 0 FirstWrite -1}
		data2_10_val {Type I LastRead 0 FirstWrite -1}
		data2_11_val {Type I LastRead 0 FirstWrite -1}
		data2_12_val {Type I LastRead 0 FirstWrite -1}
		data2_13_val {Type I LastRead 0 FirstWrite -1}
		data2_14_val {Type I LastRead 0 FirstWrite -1}
		data2_15_val {Type I LastRead 0 FirstWrite -1}
		data2_16_val {Type I LastRead 0 FirstWrite -1}
		data2_17_val {Type I LastRead 0 FirstWrite -1}
		data2_18_val {Type I LastRead 0 FirstWrite -1}
		data2_19_val {Type I LastRead 0 FirstWrite -1}
		data2_20_val {Type I LastRead 0 FirstWrite -1}
		data2_21_val {Type I LastRead 0 FirstWrite -1}
		data2_22_val {Type I LastRead 0 FirstWrite -1}
		data2_23_val {Type I LastRead 0 FirstWrite -1}
		data2_24_val {Type I LastRead 0 FirstWrite -1}
		data2_25_val {Type I LastRead 0 FirstWrite -1}
		data2_26_val {Type I LastRead 0 FirstWrite -1}
		data2_27_val {Type I LastRead 0 FirstWrite -1}
		data2_28_val {Type I LastRead 0 FirstWrite -1}
		data2_29_val {Type I LastRead 0 FirstWrite -1}
		data2_30_val {Type I LastRead 0 FirstWrite -1}
		data2_31_val {Type I LastRead 0 FirstWrite -1}}}

set hasDtUnsupportedChannel 0

set PerformanceInfo {[
	{"Name" : "Latency", "Min" : "0", "Max" : "0"}
	, {"Name" : "Interval", "Min" : "1", "Max" : "1"}
]}

set PipelineEnableSignalInfo {[
]}

set Spec2ImplPortList { 
	data1_0_val { ap_none {  { data1_0_val in_data 0 8 } } }
	data1_1_val { ap_none {  { data1_1_val in_data 0 8 } } }
	data1_2_val { ap_none {  { data1_2_val in_data 0 8 } } }
	data1_3_val { ap_none {  { data1_3_val in_data 0 8 } } }
	data1_4_val { ap_none {  { data1_4_val in_data 0 8 } } }
	data1_5_val { ap_none {  { data1_5_val in_data 0 8 } } }
	data1_6_val { ap_none {  { data1_6_val in_data 0 8 } } }
	data1_7_val { ap_none {  { data1_7_val in_data 0 8 } } }
	data1_8_val { ap_none {  { data1_8_val in_data 0 8 } } }
	data1_9_val { ap_none {  { data1_9_val in_data 0 8 } } }
	data1_10_val { ap_none {  { data1_10_val in_data 0 8 } } }
	data1_11_val { ap_none {  { data1_11_val in_data 0 8 } } }
	data1_12_val { ap_none {  { data1_12_val in_data 0 8 } } }
	data1_13_val { ap_none {  { data1_13_val in_data 0 8 } } }
	data1_14_val { ap_none {  { data1_14_val in_data 0 8 } } }
	data1_15_val { ap_none {  { data1_15_val in_data 0 8 } } }
	data1_16_val { ap_none {  { data1_16_val in_data 0 8 } } }
	data1_17_val { ap_none {  { data1_17_val in_data 0 8 } } }
	data1_18_val { ap_none {  { data1_18_val in_data 0 8 } } }
	data1_19_val { ap_none {  { data1_19_val in_data 0 8 } } }
	data1_20_val { ap_none {  { data1_20_val in_data 0 8 } } }
	data1_21_val { ap_none {  { data1_21_val in_data 0 8 } } }
	data1_22_val { ap_none {  { data1_22_val in_data 0 8 } } }
	data1_23_val { ap_none {  { data1_23_val in_data 0 8 } } }
	data1_24_val { ap_none {  { data1_24_val in_data 0 8 } } }
	data1_25_val { ap_none {  { data1_25_val in_data 0 8 } } }
	data1_26_val { ap_none {  { data1_26_val in_data 0 8 } } }
	data1_27_val { ap_none {  { data1_27_val in_data 0 8 } } }
	data1_28_val { ap_none {  { data1_28_val in_data 0 8 } } }
	data1_29_val { ap_none {  { data1_29_val in_data 0 8 } } }
	data1_30_val { ap_none {  { data1_30_val in_data 0 8 } } }
	data1_31_val { ap_none {  { data1_31_val in_data 0 8 } } }
	data2_0_val { ap_none {  { data2_0_val in_data 0 8 } } }
	data2_1_val { ap_none {  { data2_1_val in_data 0 8 } } }
	data2_2_val { ap_none {  { data2_2_val in_data 0 8 } } }
	data2_3_val { ap_none {  { data2_3_val in_data 0 8 } } }
	data2_4_val { ap_none {  { data2_4_val in_data 0 8 } } }
	data2_5_val { ap_none {  { data2_5_val in_data 0 8 } } }
	data2_6_val { ap_none {  { data2_6_val in_data 0 8 } } }
	data2_7_val { ap_none {  { data2_7_val in_data 0 8 } } }
	data2_8_val { ap_none {  { data2_8_val in_data 0 8 } } }
	data2_9_val { ap_none {  { data2_9_val in_data 0 8 } } }
	data2_10_val { ap_none {  { data2_10_val in_data 0 8 } } }
	data2_11_val { ap_none {  { data2_11_val in_data 0 8 } } }
	data2_12_val { ap_none {  { data2_12_val in_data 0 8 } } }
	data2_13_val { ap_none {  { data2_13_val in_data 0 8 } } }
	data2_14_val { ap_none {  { data2_14_val in_data 0 8 } } }
	data2_15_val { ap_none {  { data2_15_val in_data 0 8 } } }
	data2_16_val { ap_none {  { data2_16_val in_data 0 8 } } }
	data2_17_val { ap_none {  { data2_17_val in_data 0 8 } } }
	data2_18_val { ap_none {  { data2_18_val in_data 0 8 } } }
	data2_19_val { ap_none {  { data2_19_val in_data 0 8 } } }
	data2_20_val { ap_none {  { data2_20_val in_data 0 8 } } }
	data2_21_val { ap_none {  { data2_21_val in_data 0 8 } } }
	data2_22_val { ap_none {  { data2_22_val in_data 0 8 } } }
	data2_23_val { ap_none {  { data2_23_val in_data 0 8 } } }
	data2_24_val { ap_none {  { data2_24_val in_data 0 8 } } }
	data2_25_val { ap_none {  { data2_25_val in_data 0 8 } } }
	data2_26_val { ap_none {  { data2_26_val in_data 0 8 } } }
	data2_27_val { ap_none {  { data2_27_val in_data 0 8 } } }
	data2_28_val { ap_none {  { data2_28_val in_data 0 8 } } }
	data2_29_val { ap_none {  { data2_29_val in_data 0 8 } } }
	data2_30_val { ap_none {  { data2_30_val in_data 0 8 } } }
	data2_31_val { ap_none {  { data2_31_val in_data 0 8 } } }
}
