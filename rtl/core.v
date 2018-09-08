

module core (
  clk,
  rst,
  
  rdy,
  forward,
  backward, 
  x,
  e,

  vld,
  y
  );
  
  // system
  input wire clk;
  input wire rst;
  
  // input
  input wire rdy;
  input wire forward;
  input wire backward;
  input wire [15:0] x;
  input wire [15:0] e;

  // output
  output vld;
  output [15:0] y;

  // memory
  reg [15:0] w [0:99];
  reg [6:0] index;

  always @(*) begin
  
    if (forward) begin
      y = y + (w[index] * x);
    end 
    
    else if (backward) begin
      w[index] = w[index] + (e * x);
    end
    
  end
  
  always @posedge(clk) begin 
    if (rst) begin
      index = 0;
      vld = 0;
      y = 0;
    end
    else begin
    end
    
  end
