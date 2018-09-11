
verilog_depends = test.v core.v

vpi_depends = sim_vpi.c

all: vpi_compile verilog_compile

vpi_compile: $(vpi_depends)
	iverilog-vpi $(vpi_depends) -mfp16-format

verilog_compile: $(verilog_depends)
	iverilog -o sim_vpi.vvp $(verilog_depends)

run:
	vvp -M. -m sim_vpi sim_vpi.vvp

clean:
	rm -f *.vvp *.vpi *.o *.out *.vcd
