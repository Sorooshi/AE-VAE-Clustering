import glob
import subprocess

# n_repeats = 10  # SEFNAC
n_repeats = 5  # AE/VAE

if __name__ == "__main__":

    small = False
    small_vae = False

    medium = False
    medium_vae = False

    lawyers = False
    lawyers_vae = False

    org_cons = False
    org_cons_vae = False

    world_trade = False
    world_trade_vae = False

    parliament = False
    parliament_vae = False

    hvr = False
    hvr_vae = True

    cora_vae = False

    facebook_0_vae = False

    if small is True:

        # loading files
        EDGES = glob.glob('cesna_small_data/*.edges')  
        FEATURES = glob.glob('cesna_small_data/*.nodefeat')

        if len(EDGES) != len(FEATURES):
            print("Some thing is wrong")
            print("  ")
        else:
            print("everything is okay, Medium,", len(EDGES))
            print("  ")

        # running the "./cesna" program
        for i in range(len(EDGES)):
            print("edges       :", 'cesna_small_data/' + str(i + 1) + '.edges')

            print("features    :", 'cesna_small_data/' + str(i + 1) + '.nodefeat')

            print("sudo ./cesna " + "-c:5 " + "-o:results_small/small_" + str(i + 1) +
                  " -i:" + "./" + 'cesna_small_data/' + str(i + 1) + '.edges' +
                  " -a:" + "./" + 'cesna_small_data/' + str(i + 1) + '.nodefeat')

            process = subprocess.Popen(["sudo ./cesna " + "-c:5 -o:results_small/small_" + str(i + 1) +
                                        " -i:" + "./cesna_small_data/" + str(i + 1) + '.edges' +
                                        " -a:" + "./cesna_small_data/" + str(i + 1) + '.nodefeat' " - n: "
                                        + "-sa:0.005" + "-sb:0.1"],
                                       stdout=subprocess.PIPE, shell=True)

            proc_stdout = process.communicate()[0].strip()
            print(proc_stdout)
            print(" ")

    elif medium is True:

        # loading files
        EDGES = glob.glob('cesna_medium_data/*.edges')
        FEATURES = glob.glob('cesna_medium_data/*.nodefeat')

        if len(EDGES) != len(FEATURES):
            print("Some thing is wrong")
            print("  ")
        else:
            print("everything is okay, Medium,", len(EDGES))
            print("  ")

        # running the "./cesna" program
        for i in range(len(EDGES)):
            print("edges       :", 'cesna_medium_data/' + str(i + 1) + '.edges')

            print("features    :", 'cesna_medium_data/' + str(i + 1) + '.nodefeat')

            print("sudo ./cesna " + "-c:15 " + "-o:results_medium/medium_" + str(i + 1) +
                  " -i:" + "./" + 'cesna_medium_data/' + str(i + 1) + '.edges' +
                  " -a:" + "./" + 'cesna_medium_data/' + str(i + 1) + '.nodefeat')

            process = subprocess.Popen(["sudo ./cesna " + "-c:15 -o:results_medium/medium_" + str(i + 1) +
                                        " -i:" + "./cesna_medium_data/" + str(i + 1) + '.edges' +
                                        " -a:" + "./cesna_medium_data/" + str(i + 1) + '.nodefeat' " - n: "],
                                       stdout=subprocess.PIPE, shell=True)

            proc_stdout = process.communicate()[0].strip()

            print(proc_stdout)

            print(" ")

    elif medium_vae is True:

        # loading files
        EDGES = glob.glob('cesna_medium_vae_split/*.edges')
        FEATURES = glob.glob('cesna_medium_vae_split/*.nodefeat')

        if len(EDGES) != len(FEATURES):
            print("Some thing is wrong")
            print("  ")
        else:
            print("everything is okay, Medium,", len(EDGES))
            print("  ")

        # running the "./cesna" program
        for i in range(len(EDGES)):
            print("edges       :", 'cesna_medium_vae_split/' + str(i + 1) + '.edges')

            print("features    :", 'cesna_medium_vae_split/' + str(i + 1) + '.nodefeat')

            print("sudo ./cesna " + "-c:15 " + "-o:results_medium_vae/medium_" + str(i + 1) +
                  " -i:" + "./" + 'cesna_medium_vae_split/' + str(i + 1) + '.edges' +
                  " -a:" + "./" + 'cesna_medium_vae_split/' + str(i + 1) + '.nodefeat')

            process = subprocess.Popen(["sudo ./cesna " + "-c:15 -o:results_medium_vae/medium_" + str(i + 1) +
                                        " -i:" + "./cesna_medium_vae_split/" + str(i + 1) + '.edges' +
                                        " -a:" + "./cesna_medium_vae_split/" + str(i + 1) + '.nodefeat' " - n: "],
                                       stdout=subprocess.PIPE, shell=True)

            proc_stdout = process.communicate()[0].strip()

            print(proc_stdout)

            print(" ")

    elif lawyers is True:

        for i in range(n_repeats):

            process = subprocess.Popen(["sudo ./cesna " + "-c:6 -o:Lawyers" + str(i) + 
                                        " -i:" + "./EL_friends_snap" + '.edges' +
                                        " -a:" + "./EL_features_snap" + '.nodefeat' " - n: "],
                                       stdout=subprocess.PIPE, shell=True)

            proc_stdout = process.communicate()[0].strip()

            """
            with open("SIAN_Lawyers_results" + str(i) + ".txt", 'wb') as fp:
                fp.write(proc_stdout)

            print("proc_stdout:", type(proc_stdout))
            print(" ")
            """

    elif lawyers_vae is True:

        # loading files
        EDGES = glob.glob('lawyers_vae/*.edges')
        FEATURES = glob.glob('lawyers_vae/*.nodefeat')

        if len(EDGES) != len(FEATURES):
            print("Some thing is wrong")
            print("  ")
        else:
            print("everything is okay,", len(EDGES))
            print("  ")

        # running the "./cesna" program
        for i in range(len(EDGES)):

            print("edges       :", 'lawyers_vae/' + str(i + 1) + '.edges')

            print("features    :", 'lawyers_vae/' + str(i + 1) + '.nodefeat')

            print("sudo ./cesna " + "-c:6 " + "-o:lawyers_vae/lawyers_" + str(i + 1) +
                  " -i:" + "./" + 'lawyers_vae/' + str(i + 1) + '.edges' +
                  " -a:" + "./" + 'lawyers_vae/' + str(i + 1) + '.nodefeat')

            process = subprocess.Popen(["sudo ./cesna " + "-c:6 -o:lawyers_vae/lawyers_" + str(i + 1) +
                                        " -i:" + "./lawyers_vae/" + str(i + 1) + '.edges' +
                                        " -a:" + "./lawyers_vae/" + str(i + 1) + '.nodefeat' " - n: "],
                                       stdout=subprocess.PIPE, shell=True)

            proc_stdout = process.communicate()[0].strip()
            print(proc_stdout)
            print(" ")

    elif hvr is True:

        for i in range(n_repeats):

            process = subprocess.Popen(["sudo ./cesna " + "-c:2 -o:hvr" + str(i) +
                                        " -i:" + "./hvr_nets_snap" + '.edges' +
                                        " -a:" + "./hvr_features_snap" + '.nodefeat' " - n: "],
                                       stdout=subprocess.PIPE, shell=True)

            proc_stdout = process.communicate()[0].strip()

    elif hvr_vae is True:

        # loading files
        EDGES = glob.glob('hvr_vae/*.edges')
        FEATURES = glob.glob('hvr_vae/*.nodefeat')

        if len(EDGES) != len(FEATURES):
            print("Some thing is wrong")
            print("  ")
        else:
            print("everything is okay,", len(EDGES))
            print("  ")

        # running the "./cesna" program
        for i in range(len(EDGES)):
            print("edges       :", 'hvr_vae/' + str(i + 1) + '.edges')

            print("features    :", 'hvr_vae/' + str(i + 1) + '.nodefeat')

            print("sudo ./cesna " + "-c:2 " + "-o:hvr_vae/hvr_" + str(i + 1) +
                  " -i:" + "./" + 'hvr_vae/' + str(i + 1) + '.edges' +
                  " -a:" + "./" + 'hvr_vae/' + str(i + 1) + '.nodefeat')

            process = subprocess.Popen(["sudo ./cesna " + "-c:2 -o:hvr_vae/hvr_" + str(i + 1) +
                                        " -i:" + "./hvr_vae/" + str(i + 1) + '.edges' +
                                        " -a:" + "./hvr_vae/" + str(i + 1) + '.nodefeat' " - n: "],
                                       stdout=subprocess.PIPE, shell=True)

            proc_stdout = process.communicate()[0].strip()
            print(proc_stdout)
            print(" ")

    elif org_cons is True:

        for i in range(n_repeats):

            process = subprocess.Popen(["sudo ./cesna " + "-c:2 -o:org_cons" + str(i) +
                                        " -i:" + "./org_consult_nets_snap" + '.edges' +
                                        " -a:" + "./org_consult_features_snap" + '.nodefeat' " - n: "],
                                       stdout=subprocess.PIPE, shell=True)

            proc_stdout = process.communicate()[0].strip()

    elif world_trade is True:

        for i in range(n_repeats):
            process = subprocess.Popen(["sudo ./cesna " + "-c:15 -o:world_trade" + str(i) +
                                        " -i:" + "./WT_net_snap" + '.edges' +
                                        " -a:" + "./WT_features_snap" + '.nodefeat' " - n: "],
                                       stdout=subprocess.PIPE, shell=True)

            proc_stdout = process.communicate()[0].strip()

    elif world_trade_vae is True:

        # loading files
        EDGES = glob.glob('world_trade_vae/*.edges')
        FEATURES = glob.glob('world_trade_vae/*.nodefeat')

        if len(EDGES) != len(FEATURES):
            print("Some thing is wrong")
            print("  ")
        else:
            print("everything is okay,", len(EDGES))
            print("  ")

        # running the "./cesna" program
        for i in range(len(EDGES)):
            print("edges       :", 'world_trade_vae/' + str(i + 1) + '.edges')

            print("features    :", 'world_trade_vae/' + str(i + 1) + '.nodefeat')

            print("sudo ./cesna " + "-c:15 " + "-o:world_trade_vae/world_trade_" + str(i + 1) +
                  " -i:" + "./" + 'world_trade_vae/' + str(i + 1) + '.edges' +
                  " -a:" + "./" + 'world_trade_vae/' + str(i + 1) + '.nodefeat')

            process = subprocess.Popen(["sudo ./cesna " + "-c:15 -o:world_trade_vae/world_trade_" + str(i + 1) +
                                        " -i:" + "./world_trade_vae/" + str(i + 1) + '.edges' +
                                        " -a:" + "./world_trade_vae/" + str(i + 1) + '.nodefeat' " - n: "],
                                       stdout=subprocess.PIPE, shell=True)

            proc_stdout = process.communicate()[0].strip()
            print(proc_stdout)
            print(" ")

    elif parliament is True:

        for i in range(n_repeats):
            process = subprocess.Popen(["sudo ./cesna " + "-c:7 -o:parliament" + str(i) +
                                        " -i:" + "./Parl_nets_snap" + '.edges' +
                                        " -a:" + "./Parl_features_snap" + '.nodefeat' " - n: "],
                                       stdout=subprocess.PIPE, shell=True)

            proc_stdout = process.communicate()[0].strip()

    elif parliament_vae is True:

        # loading files
        EDGES = glob.glob('parliament_vae/*.edges')
        FEATURES = glob.glob('parliament_vae/*.nodefeat')

        if len(EDGES) != len(FEATURES):
            print("Some thing is wrong")
            print("  ")
        else:
            print("everything is okay,", len(EDGES))
            print("  ")

        # running the "./cesna" program
        for i in range(len(EDGES)):
            print("edges       :", 'parliament_vae/' + str(i + 1) + '.edges')

            print("features    :", 'parliament_vae/' + str(i + 1) + '.nodefeat')

            print("sudo ./cesna " + "-c:7 " + "-o:parliament_vae/parliament_" + str(i + 1) +
                  " -i:" + "./" + 'parliament_vae/' + str(i + 1) + '.edges' +
                  " -a:" + "./" + 'parliament_vae/' + str(i + 1) + '.nodefeat')

            process = subprocess.Popen(["sudo ./cesna " + "-c:7 -o:parliament_vae/parliament_" + str(i + 1) +
                                        " -i:" + "./parliament_vae/" + str(i + 1) + '.edges' +
                                        " -a:" + "./parliament_vae/" + str(i + 1) + '.nodefeat' " - n: "],
                                       stdout=subprocess.PIPE, shell=True)

            proc_stdout = process.communicate()[0].strip()
            print(proc_stdout)
            print(" ")

    elif cora_vae is True:

        # loading files
        EDGES = glob.glob('cora_vae/*.edges')
        FEATURES = glob.glob('cora_vae/*.nodefeat')

        if len(EDGES) != len(FEATURES):
            print("Some thing is wrong")
            print("  ")
        else:
            print("everything is okay,", len(EDGES))
            print("  ")

        # running the "./cesna" program
        for i in range(len(EDGES)):
            print("edges       :", 'cora_vae/' + str(i + 1) + '.edges')

            print("features    :", 'cora_vae/' + str(i + 1) + '.nodefeat')

            print("sudo ./cesna " + "-c:7 " + "-o:cora_vae/cora_" + str(i + 1) +
                  " -i:" + "./" + 'cora_vae/' + str(i + 1) + '.edges' +
                  " -a:" + "./" + 'cora_vae/' + str(i + 1) + '.nodefeat')

            process = subprocess.Popen(["sudo ./cesna " + "-c:7 -o:cora_vae/cora_" + str(i + 1) +
                                        " -i:" + "./cora_vae/" + str(i + 1) + '.edges' +
                                        " -a:" + "./cora_vae/" + str(i + 1) + '.nodefeat' " - n: "],
                                       stdout=subprocess.PIPE, shell=True)

            proc_stdout = process.communicate()[0].strip()
            print(proc_stdout)
            print(" ")

    elif facebook_0_vae is True:

        # loading files
        EDGES = glob.glob('facebook_0_vae/*.edges')
        FEATURES = glob.glob('facebook_0_vae/*.nodefeat')

        if len(EDGES) != len(FEATURES):
            print("Some thing is wrong")
            print("  ")
        else:
            print("everything is okay,", len(EDGES))
            print("  ")

        # running the "./cesna" program
        for i in range(len(EDGES)):
            print("edges       :", 'facebook_0_vae/' + str(i + 1) + '.edges')

            print("features    :", 'facebook_0_vae/' + str(i + 1) + '.nodefeat')

            print("sudo ./cesna " + "-c:10 " + "-o:facebook_0_vae/facebook_0_" + str(i + 1) +
                  " -i:" + "./" + 'facebook_0_vae/' + str(i + 1) + '.edges' +
                  " -a:" + "./" + 'facebook_0_vae/' + str(i + 1) + '.nodefeat')

            process = subprocess.Popen(["sudo ./cesna " + "-c:10 -o:facebook_0_vae/facebook_0_" + str(i + 1) +
                                        " -i:" + "./facebook_0_vae/" + str(i + 1) + '.edges' +
                                        " -a:" + "./facebook_0_vae/" + str(i + 1) + '.nodefeat' " - n: "],
                                       stdout=subprocess.PIPE, shell=True)

            proc_stdout = process.communicate()[0].strip()
            print(proc_stdout)
            print(" ")



