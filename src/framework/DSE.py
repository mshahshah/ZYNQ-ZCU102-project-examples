
from dnn_tools import *




def directive_converter(argument):
    switcher = {
        'inline': "set_directive_inline",
        'pipeline': "set_directive_pipeline",
        'unroll': "set_directive_unroll",
        'reshape': "set_directive_array_reshape",
        'resource': "set_directive_resource",
        'bram': "RAM_1P_BRAM",
        'lutram': "RAM_1P_LUTRAM",
        'flushing': "enable_flush",
        'complete': "complete",
        'cyclic': "cyclic",
        '': ""
    }
    if switcher.get(argument) is None:
        print("Error: Invalid Directive for label  \'" + argument + '\'')
        print("Compilation stopped due to an error in directive list, check the directive/option format")
        exit()
        return argument
    else:
        return switcher.get(argument)


def HLS_directive_generator(request, mode):
    directives = []
    if (mode == 'changing'):
        directives.append('')

    temp = ''
    for option in request['option'].split(','):
        option = directive_converter(option.lower())
        for directive in request['directive'].split(','):
            directive = directive_converter(directive.lower())
            if (directive == 'set_directive_unroll'):
                if (option == ''):
                    temp = 'set_directive_unroll' + ' \"' + request['name'] + '\"'
                else:
                    temp = 'set_directive_unroll' + ' \"' + request['name'] + '\"' + ' -factor' + request['factor']

                directives.append(temp)

            elif (directive == 'set_directive_pipeline'):
                if ((option == '') & (request['factor'] == '')):
                    temp = 'set_directive_pipeline' + ' \"' + request['name'] + '\"'
                elif ((option == '') & (request['factor'] != '')):
                    temp = 'set_directive_pipeline' + ' -II ' + request['factor'] + ' \"' + request['name'] + '\"'
                elif ((option != '') & (request['factor'] == '')):
                    temp = 'set_directive_pipeline' + ' -' + option + ' \"' + request['name'] + '\"'
                elif ((option != '') & (request['factor'] != '')):
                    temp = 'set_directive_pipeline' + ' -II ' + request['factor'] + ' -' + option + ' \"' + request[
                        'name'] + '\"'

                directives.append(temp)

            elif (directive == 'set_directive_inline'):
                temp = 'set_directive_inline' + ' \"' + request['name'] + '\"'
                directives.append(temp)

            elif (directive == 'set_directive_array_reshape'):
                array_name = request['name'].split('/')
                factors = request['factor'].split(',')
                if (option == 'complete'):
                    for factor in factors:
                        temp = 'set_directive_array_reshape' + ' -type complete' + ' -dim ' + factor + ' \"' + \
                               array_name[0] + '\" ' + array_name[1]
                        directives.append(temp)
                elif (option == 'cyclic'):
                    for factor in factors:
                        temp = 'set_directive_array_reshape' + ' -type cyclic' + ' -dim ' + factor + ' \"' + array_name[
                            0] + '\" ' + array_name[1]
                        directives.append(temp)

            elif (directive == 'set_directive_resource'):
                array_name = request['name'].split('/')
                # options = request['option'].split(',')
                # for option in options:
                temp = 'set_directive_resource' + ' -core ' + option + ' \"' + array_name[0] + '\" ' + array_name[1]
                directives.append(temp)
    return directives


def extract_data(cfg, fpga_chip, solutions_list, XML_list):
    summary = {}
    summary['HLS'] = {}
    summary['PR'] = {}
    LATENCY_BEST = []
    LATENCY_WORST = []
    AREA_BRAM = []
    AREA_FF = []
    AREA_LUT = []
    AREA_DSP = []
    AREA_DSP_norm = []
    AREA_FF_norm = []
    AREA_LUT_norm = []
    AREA_BRAM_norm = []
    TIMING = []
    CLOCK = []
    for solution in solutions_list:
        TEMP = solution["ModuleInfo"]["Metrics"][cfg.dse_tool.topmodule]
        LATENCY_BEST.append(int(TEMP["Latency"]["LatencyBest"]))
        LATENCY_WORST.append(int(TEMP["Latency"]["LatencyWorst"]))
        TIMING.append(float(TEMP["Timing"]["Estimate"]))
        CLOCK.append(float(TEMP["Timing"]["Target"]))
        AREA_DSP.append(int(TEMP["Area"]["DSP48E"]))
        AREA_FF.append(int(TEMP["Area"]["FF"]))
        AREA_LUT.append(int(TEMP["Area"]["LUT"]))
        AREA_BRAM.append(int(TEMP["Area"]["BRAM_18K"]))
        AREA_DSP_norm.append(int(TEMP["Area"]["DSP48E"]) * 100 / fpga_chip.Resources.DSP48E)
        AREA_FF_norm.append(int(TEMP["Area"]["FF"]) * 100 / fpga_chip.Resources.FF)
        AREA_LUT_norm.append(int(TEMP["Area"]["LUT"]) * 100 / fpga_chip.Resources.LUT)
        AREA_BRAM_norm.append(int(TEMP["Area"]["BRAM_18K"]) * 100 / fpga_chip.Resources.BRAM_18K)

    summary['HLS']['LATENCY_BEST'] = LATENCY_BEST
    summary['HLS']['LATENCY_WORST'] = LATENCY_WORST
    summary['HLS']['TIMING'] = TIMING
    summary['HLS']['CLOCK'] = CLOCK
    summary['HLS']['AREA_DSP'] = AREA_DSP
    summary['HLS']['AREA_FF'] = AREA_FF
    summary['HLS']['AREA_LUT'] = AREA_LUT
    summary['HLS']['AREA_BRAM'] = AREA_BRAM

    summary['HLS']['DSP_NORM'] = AREA_DSP_norm
    summary['HLS']['FF_NORM'] = AREA_FF_norm
    summary['HLS']['LUT_NORM'] = AREA_LUT_norm
    summary['HLS']['BRAM_NORM'] = AREA_BRAM_norm
    TOTAL_NORM = [AREA_DSP_norm[i] + AREA_FF_norm[i] + AREA_LUT_norm[i] + AREA_BRAM_norm[i] for i in
                  range(len(AREA_FF_norm))]

    summary['HLS']['TOTAL_NORM'] = TOTAL_NORM

    for item in XML_list.keys():
        summary['PR'][item] = XML_list[item]
    return summary


class dse(object):
    def __init__(self,cfg):
        self.path = cfg.path
        self.tclfile_template = {}

    def PickRandomCombinations(self, cfg, Max_Num_of_combinations):
        Num_of_combinations = round(cfg.dse_tool.TrainingPercentage * Max_Num_of_combinations / 100)
        print("Number of randomly selected Combinations = " + str(Num_of_combinations))
        Training_combination_selection = np.random.choice(range(Max_Num_of_combinations), Num_of_combinations,
                                                          replace=False).tolist()
        Testing_combination_selection = np.delete(np.arange(Max_Num_of_combinations),
                                                  Training_combination_selection).tolist()
        return Training_combination_selection, Testing_combination_selection

    def create_directory(self, cfg):
        if not os.path.exists(self.path.design_dse):
            os.makedirs(self.path.design_dse)

        if not os.path.exists(self.path.hls_run):
            os.makedirs(self.path.hls_run)

    def copy_run_files(self, cfg):
        for file in cfg.dse_tool.files_be_copied:
            src = self.path.design + '/' + file
            trg = self.path.hls_run + '/' + file
            copyfile(src, trg)
        file = "vivado_hls_cmd.bat"
        src = self.path.tool + "/" + file
        trg = self.path.hls_run + '/' + file
        copyfile(src, trg)

    def create_script_tcl(self, cfg):
        attr_lines = []

        attr_lines.append("open_project hls_prj")
        attr_lines.append("set_top " + cfg.dse_tool.topmodule)
        for file in cfg.dse_tool.files_be_copied:
            attr_lines.append("add_files " + file)

        attr_lines.append("open_solution tmp_sol")
        attr_lines.append(cfg.FPGA.set_part)
        attr_lines.append("create_clock -period {} -name default".format(cfg.FPGA.clock_period))
        attr_lines.append("set_clock_uncertainty 1")

        attr_lines.append("for {set sol 0} {$sol < 3} {incr sol} {")
        attr_lines.append("   set directive_file [format \"directive_%d.tcl\" $sol]")
        attr_lines.append("   set json_file [format \"solution_%d.json\" $sol]")
        attr_lines.append("   puts \"-------------------------------------------------------\" ")
        attr_lines.append("   puts \"synthesizing using this directive : $directive_file\" ")
        attr_lines.append("   puts \"-------------------------------------------------------\" ")
        attr_lines.append("   source $directive_file")
        attr_lines.append("   csynth_design")
        if (cfg.dse_tool.run_implementation):
            attr_lines.append("   export_design -flow impl -rtl verilog -format ip_catalog")

        attr_lines.append("   file copy -force hls_prj/tmp_sol/tmp_sol_data.json ../HLS_versions/$json_file")
        attr_lines.append("}")
        attr_lines.append("exit")

        f = open(self.path.hls_run + "/script.tcl", "w+")
        for attr_line in attr_lines:
            f.write("%s\n" % attr_line)
        f.close()

    def create_directive_tcl(self, directive_list_comb, directive_list_fix, solution_num):
        prj_path = self.path.hls_run + "/"
        f = open(prj_path + "directive_%d.tcl" % solution_num, "w+")
        temp = []
        temp.append("\n######################\n# below are fixed directives \n\n")
        for Vname in directive_list_fix.keys():
            for directive in directive_list_fix[Vname]:
                temp.append("%s\n" % directive)

        temp.append("# below are variable directives \n")
        for j in range(len(directive_list_comb)):
            # f.write("%s\n" % directive_list_comb[j][solution_num])
            temp.append("%s\n" % directive_list_comb[j][solution_num])
        for line in temp:
            f.write(line)
        f.write('\n')
        f.close()
        return temp

    def pars_directive_list(self, cfg):
        file = self.path.tool + '/' + cfg.dse_tool.list_of_combinations
        input_file = csv.DictReader(open(file))
        dict_list = []
        for line in input_file:
            dict_list.append(line)

        Invariante_list = []
        Changing_list = []
        DataType_list = cfg.FPGA.data_type
        ClockPeriod_list = cfg.FPGA.clock_period

        for i in range(len(dict_list)):
            if (dict_list[i]['use'].lower() == 'YES'.lower()):
                if (dict_list[i]['type'] == 'Invariante'):
                    Invariante_list.append({key: dict_list[i][key] for key in (dict_list[i].keys() - ['use', 'type'])})
                elif (dict_list[i]['type'] == 'Changing'):
                    Changing_list.append({key: dict_list[i][key] for key in (dict_list[i].keys() - ['use', 'type'])})
                elif (dict_list[i]['type'] == 'CLKPERIOD'):
                    ClockPeriod_list = [key for key in (dict_list[i]['name'].split(','))]
                elif (cfg.dse_tool.has_header_typedef):
                    DataType_list = [key for key in (dict_list[i]['name'].split(','))]

        Invariante_directive_list = {}
        for i in range(len(Invariante_list)):
            Invariante_directive_list.update(
                {Invariante_list[i]['name']: HLS_directive_generator(Invariante_list[i], 'invariante')})
        print("Invariante Directives are compiled")
        Changing_directive_list = {}
        for i in range(len(Changing_list)):
            Changing_directive_list.update(
                {Changing_list[i]['name']: HLS_directive_generator(Changing_list[i], 'changing')})
        print("Changing Directives are compiled")

        return Invariante_directive_list, Changing_directive_list, DataType_list, ClockPeriod_list

    def create_benchmark_combinations(self, directive_list_fix, directive_list):

        len_combination = 1
        for lable in directive_list.keys():
            len_combination = len_combination * len(directive_list[lable])

        length_dict = {key: len(value) for key, value in directive_list.items()}

        directive_list_comb = []
        f = 1
        for lable in directive_list.keys():
            temp = []
            rep = int(len_combination / (f * length_dict[lable]))
            for i in range(f):
                for directive in directive_list[lable]:
                    for k in range(rep):
                        temp.append(directive)
            f = f * length_dict[lable]
            directive_list_comb.append(temp)
        return directive_list_fix, directive_list_comb

    def synthesis_design(self):
        prj_path = os.getcwd()
        os.chdir(self.path.hls_run)
        os.system("vivado_hls_cmd.bat")
        # subprocess.Popen("vivado_hls_cmd.bat")
        os.chdir(prj_path)

    def load_solutions(self, cfg, Training_combination_selection):
        solutions_list = []
        XML_list = {}
        CLB_list = []
        LUT_list = []
        FF_list = []
        DSP_list = []
        BRAM_list = []
        Solution_num = []
        solution = 0
        os.chdir(self.path.design_dse)
        for file in glob.glob("*.json"):
            try:
                with open(file) as json_data:
                    d = json.load(json_data)
                    solutions_list.append(d)
                    json_data.close()
            except IOError:
                print("I/O Error, solution{}_Syn.json is not exist".format(solution))
            if cfg.dse_tool.run_implementation:
                xml_path = "./{}/solution{}_PR.xml".format(self.path.design_dse, solution)
                mydoc = minidom.parse(xml_path)
                Solution_num.append(Training_combination_selection[solution])
                CLB_list.append((mydoc.getElementsByTagName('CLB')[0]).firstChild.data)
                LUT_list.append(mydoc.getElementsByTagName('LUT')[0].firstChild.data)
                FF_list.append(mydoc.getElementsByTagName('FF')[0].firstChild.data)
                DSP_list.append(mydoc.getElementsByTagName('DSP')[0].firstChild.data)
                BRAM_list.append(mydoc.getElementsByTagName('BRAM')[0].firstChild.data)
                XML_list['CLB'] = CLB_list
                XML_list['LUT'] = LUT_list
                XML_list['FF'] = FF_list
                XML_list['DSP'] = DSP_list
                XML_list['BRAM'] = BRAM_list
                XML_list['Solution_num'] = Solution_num
                solution = solution + 1
        os.chdir(self.path.tool)
        return solutions_list, XML_list

    def create_text_report(self, cfg, executed_directives, performance_summary, DataType_list_used, solutions_list,
                           Synthesized_combination_selection):
        f = open(self.path.design_dse + '/' + "Summary_" + cfg.dse_tool.dse_name + ".txt", "w+")
        len_solutions = len(solutions_list)
        for solution in range(len_solutions):
            f.write('\n**************  Solution Number = {} *************** \n'.format(
                Synthesized_combination_selection[solution]))
            f.write('\nHLS results:  ')
            for item in performance_summary['HLS'].keys():
                f.write(item + " = {} , ".format(performance_summary['HLS'][item][solution]))
            f.write('\nPost Implementation results:  ')
            for item in performance_summary['PR'].keys():
                f.write(item + " = , ".format(performance_summary['PR'][item][solution]))
            f.write('\n\n')

            f.write("Data type used = {}".format(DataType_list_used[solution]))
            f.write('\n')

            for line in executed_directives[solution]:
                f.write(line)
            f.write("------------------------------------------------------------------\n")
        f.write('\n')
        f.close()
        print("summary is made")

    def create_excel_report(self, cfg, executed_directives, performance_summary, DataType_list_used, solutions_list,
                            Synthesized_combination_selection):
        f = open(self.path.design_dse + '/' + "Summary_" + cfg.dse_tool.dse_name + ".csv", "w+")
        len_solutions = len(solutions_list)
        f.write('solution,' + 'Data_type,')
        for item in performance_summary['HLS'].keys():
            f.write('HLS_' + item + ',')
        for item in performance_summary['PR'].keys():
            f.write('PR_' + item + ',')
        f.write('\n')
        for solution in range(len_solutions):
            f.write(str(Synthesized_combination_selection[solution]))
            f.write(',')
            f.write(DataType_list_used[solution])
            for item in performance_summary['HLS'].keys():
                f.write("," + str(performance_summary['HLS'][item][solution]))
            for item in performance_summary['PR'].keys():
                f.write("," + str(performance_summary['PR'][item][solution]))
            f.write('\n')
        f.close()
        print("summary is made")

    def insert_header_file_line(self, cfg, index, header_arguments):
        in_file = open(cfg.dse_tool.dse_name + "/benchmark.h", "r")
        contents = in_file.readlines()
        in_file.close()
        contents.insert(index, '#define ' + header_arguments + '\n')

        in_file = open("hls_prj/" + "benchmark.h", "w")
        contents = "".join(contents)
        in_file.write(contents)
        in_file.close()

    def WriteDictToCSV(csv_file, csv_columns, dict_data):
        with open(csv_file, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for data in dict_data:
                writer.writerow(data)
        return
    ## -------------------------------------------------------------------------------------------- ##

cfg = gen_configs()
cfg.parse_input_arguments('input_arguments.yaml')

input_args = sys.argv[1:]
if (len(input_args)) == 1:
    cfg.dse_tool.dse_name = input_args[0]
elif (len(input_args)) == 2:
    cfg.dse_tool.dse_name = input_args[0]
    if input_args[1] == 'syn':
        cfg.dse_tool.run_synthesis = True
    else:
        cfg.dse_tool.run_synthesis = False
elif (len(input_args)) == 3:
    cfg.dse_tool.dse_name = input_args[0]
    if input_args[1] == 'syn':
        cfg.dse_tool.run_synthesis = True
    else:
        cfg.dse_tool.run_synthesis = False

    if input_args[2] == 'impl':
        cfg.dse_tool.run_implementation = True
    else:
        cfg.dse_tool.run_implementation = False

cfg.create_paths()

#cfg = parse_input_arguments('input_arguments.yaml')



fpga_chips = load_fpga_info('fpga_resources.yaml')
fpga_chip = fpga_chips['FPGA_A']
AC = dse(cfg)

# AC.goto_prj_dir(cfg)
Invariante_directive_list, Changing_directive_list, DataType_list, ClockPeriod_list = AC.pars_directive_list(cfg)
[directive_list_fix, directive_list_comb] = AC.create_benchmark_combinations(Invariante_directive_list,
                                                                             Changing_directive_list)

Num_of_combinations = len(directive_list_comb[0])

if (cfg.dse_tool.run_synthesis):
    AC.create_directory(cfg)
    AC.copy_run_files(cfg)
    Training_combination_selection, Testing_combination_selection = AC.PickRandomCombinations(cfg, Num_of_combinations)
    Synthesized_combination_selection = []
    Save_variables(AC.path.design_dse, [Training_combination_selection, Testing_combination_selection,
                                              Synthesized_combination_selection])
else:
    Training_combination_selection, Testing_combination_selection, Synthesized_combination_selection = Load_variables(
        AC.path.design_dse)

print(
    "\n\n Number of Combinations are : " + str(Num_of_combinations * len(DataType_list) * len(ClockPeriod_list)) + '\n')
print('number of Training sets = ' + str(Training_combination_selection))
executed_directives = []
DataType_list_used = []

for CK_period in ClockPeriod_list:
    cfg.FPGA.clock_period = CK_period
    for header_arguments in DataType_list:
        for solution in Training_combination_selection:
            if (cfg.dse_tool.has_header_typedef):
                AC.insert_header_file_line(cfg, 15, header_arguments)

            executed_directives.append(AC.create_directive_tcl(directive_list_comb, directive_list_fix, solution))
            Synthesized_combination_selection.append(solution)
            DataType_list_used.append(header_arguments)

    if cfg.dse_tool.run_synthesis:
        AC.create_script_tcl(cfg)
        AC.synthesis_design()

(solutions_list, XML_list) = AC.load_solutions(cfg, Synthesized_combination_selection)
performance_summary = extract_data(cfg, fpga_chip, solutions_list, XML_list)
# AC.save_variables(cfg,[executed_directives,performance_summary])
AC.create_text_report(cfg, executed_directives, performance_summary, DataType_list_used, solutions_list,
                      Synthesized_combination_selection)
AC.create_excel_report(cfg, executed_directives, performance_summary, DataType_list_used, solutions_list,
                       Synthesized_combination_selection)
print('khelas shod')
