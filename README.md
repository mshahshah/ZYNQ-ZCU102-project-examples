We have described a Python-based automated tool for mapping CNNs on FPGAs, which helps the software developer quickly build an accelerator, analyze, simulate, verify, and customize it with various choices of FPGA chip, performance, power, and area efficiency.


The CNN design framework provides a unified software-based design flow that enables software engineers to design CNN models that will meet final implementation goals in Performance, Power, and Area (PPA).

The framework integrates and supports Xilinx Vivado HLS tool-flow to design CNN accelerator on Xilinx ZYNQ-based FPGAs. To achieve a high-performance accelerator, the framework has the following features:

- A flexible and customizable accelerator for CNN design applications,
- A fast and reliable system builder with a detailed analyses reports,
- Minimizing the number of operations,
- Arranging data in on-chip memory to maximize computation to communication ratio,
- Maximizing accelerator throughput by considering power consumption limitation,
- Maximizing parallelism by utilizing available resources within the limitations of memory bandwidth,
- Maximizing power efficiency


All phases and steps for building the accelerator are controlled in the proposed software framework and can be executed on a variety of platforms. The tool only requires the Python tool with some standard packages and a Vivado tool. The tool can be executed on a variety of platforms to organize the files, generated output project, test-data, analyzed data, project files, software code, and reports in a well-organized structure.
