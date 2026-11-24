module dvi_divide_3_clk (
    input wire clk,
    input wire rst_n,
    output wire clk_out
);

reg [1:0] counter;
reg posclk; 
reg negclk;

https://platform.claude.com/oauth/authorize?code=true&client_id=41077d10-94b8-41
94-be48-d251e9eb21b4&response_type=code&redirect_uri=https%3A%2F%2Fplatform.clau
de.com%2Foauth%2Fcode%2Fcallback&scope=user%3Aprofile+user%3Ainference&code_chal
lenge=UatpPTKr8ddGkJ26vtMoOyr6qIdBEdFlKdAjG-IfBEY&code_challenge_method=S256&sta
te=ehCy9f5QVGyHWhAzzDod0fc4XECMCFscjnUF5ksSWEs
assign clk_out = posclk | negclk;

always @(posedge clk) begin
    if (!rst_n) begin
        counter <= 2'b00;
        posclk <= 1'b0;
    end
    else begin
        case (counter)
            2'b00: begin
                counter <= 2'b01;
                posclk <= 1'b0; 
            end
            2'b01: begin
                counter <= 2'b10;
                posclk <= 1'b1; 
            end
            2'b10: begin
                counter <= 2'b00;
                posclk <= 1'b0; 
            end
            default: begin
                counter <= 2'b00;
                posclk <= 1'b0;
            end
        endcase
    end
end

always @(negedge clk) begin
    if (!rst_n) begin
        negclk <= 1'b0;
    end
    else begin
        negclk <= posclk; 
    end
end