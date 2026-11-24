// State the contract first:
// - One clock.
// - Synchronous active-low reset.
// - One input and one output transfer per cycle.
// - Any positive integer depth, including 1.
// - Small flop array with asynchronous read.
// - No empty bypass.
// - Full FIFO permits simultaneous removal and replacement.
// No transfers count during reset.



module sync_fifo #(
    parameter D = 10,
    parameter W = 8
)(
    input wire clk,
    input wire rst_n,
    input wire rd_en,
    input wire wr_en,
    input wire [W-1:0] in,
    output wire [W-1:0] out
);

localparam PTR_W = (D > 1) ? $clog2(D) : 1;

reg [W-1:0] arr [0:D-1];

// Extra MSB is the wrap bit
reg [PTR_W:0] rd_ptr;
reg [PTR_W:0] wr_ptr;

wire full, empty;
wire push, pop;

// Same address, different wrap bit = full
assign full =
    (rd_ptr[PTR_W] != wr_ptr[PTR_W]) &&
    (rd_ptr[PTR_W-1:0] == wr_ptr[PTR_W-1:0]);

// Same complete pointer = empty
assign empty = (rd_ptr == wr_ptr);

// Accepted transfers
assign pop  = rst_n && rd_en && !empty;
assign push = rst_n && wr_en && (!full || pop);

// Only use address bits to index array
assign out = arr[rd_ptr[PTR_W-1:0]];

always @(posedge clk) begin
    if (!rst_n) begin
        rd_ptr <= '0;
        wr_ptr <= '0;
    end
    else begin
        if (pop) begin
            if (rd_ptr[PTR_W-1:0] == D-1) begin
                rd_ptr[PTR_W-1:0] <= '0;
                rd_ptr[PTR_W] <= ~rd_ptr[PTR_W];
            end
            else begin
                rd_ptr[PTR_W-1:0] <=
                    rd_ptr[PTR_W-1:0] + 1'b1;
            end
        end

        if (push) begin
            arr[wr_ptr[PTR_W-1:0]] <= in;

            if (wr_ptr[PTR_W-1:0] == D-1) begin
                wr_ptr[PTR_W-1:0] <= '0;
                wr_ptr[PTR_W] <= ~wr_ptr[PTR_W];
            end
            else begin
                wr_ptr[PTR_W-1:0] <=
                    wr_ptr[PTR_W-1:0] + 1'b1;
            end
        end
    end
end

endmodule
