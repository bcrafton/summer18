

module test;

  reg [15:0] a;
  reg [15:0] b;
  reg [15:0] accum;

  initial begin
      accum = 0;
      a = 10;
      b = 20;
  
      accum = $mac(accum, a, b);
      $display("%b\n", accum);
      
      accum = $mac(accum, a, b);
      $display("%b\n", accum);
  end
  
endmodule
  
