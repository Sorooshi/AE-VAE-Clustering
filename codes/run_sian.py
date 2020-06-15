import glob
import subprocess

# n_repeats = 10  # SEFNAC
n_repeats = 5  # AE/VAE

if __name__ == "__main__":

    small = False

    medium = False
    medium_vae = False

    lawyers = False
    lawyers_vae = False

    org_cons = False

    world_trade = False
    world_trade_vae = False

    parliament = False
    parliament_vae = False

    hvr = False
    hvr_vae = False

    cora_vae = False

    facebook_0_vae = False
    
    if small is True:
        # loading files
        GMLs = glob.glob('sian_small_data/*.gml')

        print("len(GMLs):", len(GMLs))

        # running the "./metadata" program
        for i in range(len(GMLs)):
            print(".gmls       :", 'sian_small_data/small' + str(i + 1) + '.gml')
            print("sudo ./metadata < small_" + str(i) + ".gml")

            process = subprocess.Popen(["sudo ./metadata < sian_small_data/small_" + str(i + 1) + ".gml"],
                                       stdout=subprocess.PIPE, shell=True)

            proc_stdout = process.communicate()[0].strip()

            with open("sian_small_results/small_result" + str(i + 1) + ".txt", 'wb') as fp:
                fp.write(proc_stdout)

            print("proc_stdout:", type(proc_stdout))
            print(" ")

    elif medium is True:

        # loading files
        GMLs = glob.glob('sian_medium_data/*.gml')

        print("len(GMLs):", len(GMLs))

        # running the "./metadata" program
        for i in range(40, len(GMLs)):
            print(".gmls       :", 'sian_medium_data/medium' + str(i + 1) + '.gml')
            print("sudo ./metadata < medium_" + str(i) + ".gml")


            process = subprocess.Popen(["sudo ./metadata < sian_medium_data/medium_" + str(i + 1) + ".gml"],
                                       stdout=subprocess.PIPE, shell=True)

            proc_stdout = process.communicate()[0].strip()

            with open("sian_medium_results/medium_result" + str(i + 1) + ".txt", 'wb') as fp:
                fp.write(proc_stdout)

            print("proc_stdout:", type(proc_stdout))
            print(" ")

    elif medium_vae is True:

        # loading files
        GMLs = glob.glob('sian_medium_data_vae_split/*.gml')

        print("len(GMLs):", len(GMLs))

        # running the "./metadata" program
        for i in range(0, len(GMLs)):
            print(".gmls       :", 'sian_medium_data_vae_split/medium' + str(i + 1) + '.gml')
            print("sudo ./metadata < medium_" + str(i) + ".gml")

            process = subprocess.Popen(["sudo ./metadata < sian_medium_data_vae_split/medium_" + str(i + 1) + ".gml"],
                                       stdout=subprocess.PIPE, shell=True)

            proc_stdout = process.communicate()[0].strip()

            with open("sian_medium_vae_results/medium_result" + str(i + 1) + ".txt", 'wb') as fp:
                fp.write(proc_stdout)

            print("proc_stdout:", type(proc_stdout))
            print(" ")

    elif lawyers is True:

        for i in range(n_repeats):
            # loading files
            # running the "./metadata" program
            process = subprocess.Popen(["sudo ./metadata < " "Lawyers.gml"],
                                       stdout=subprocess.PIPE, shell=True)

            proc_stdout = process.communicate()[0].strip()

            with open("lawyers/SIAN_Lawyers_results" + str(i) + ".txt", 'wb') as fp:
                fp.write(proc_stdout)

            print("proc_stdout:", type(proc_stdout))
            print(" ")

    elif lawyers_vae is True:

        # loading files
        GMLs = glob.glob('lawyers_vae_sian/*.gml')

        print("len(GMLs):", len(GMLs))

        # running the "./metadata" program
        for i in range(0, len(GMLs)):
            print(".gmls       :", 'lawyers_vae_sian/' + str(i) + '.gml')
            print("sudo ./metadata < " + str(i) + ".gml")

            process = subprocess.Popen(["sudo ./metadata < lawyers_vae_sian/" + str(i) + ".gml"],
                                       stdout=subprocess.PIPE, shell=True)

            proc_stdout = process.communicate()[0].strip()

            with open("lawyers_vae_sian/" + str(i) + ".txt", 'wb') as fp:
                fp.write(proc_stdout)

            print("proc_stdout:", type(proc_stdout))
            print(" ")

    elif org_cons is True:

        for i in range(n_repeats):
            # loading files
            # running the "./metadata" program
            process = subprocess.Popen(["sudo ./metadata < " "SIAN_org_consult.gml"],
                                       stdout=subprocess.PIPE, shell=True)

            proc_stdout = process.communicate()[0].strip()

            with open("SIAN_org_consult_results" + str(i) + ".txt", 'wb') as fp:
                fp.write(proc_stdout)

            print("proc_stdout:", type(proc_stdout))
            print(" ")

    elif world_trade is True:

        for i in range(n_repeats):
            # loading files
            # running the "./metadata" program
            process = subprocess.Popen(["sudo ./metadata < " "world_trade.gml"],
                                       stdout=subprocess.PIPE, shell=True)

            proc_stdout = process.communicate()[0].strip()

            with open("SIAN_world_trade_results" + str(i) + ".txt", 'wb') as fp:
                fp.write(proc_stdout)

            print("proc_stdout:", type(proc_stdout))
            print(" ")

    elif world_trade_vae is True:

        # loading files
        GMLs = glob.glob('world_trade_vae_sian/*.gml')

        print("len(GMLs):", len(GMLs))

        # running the "./metadata" program
        for i in range(0, len(GMLs)):
            print(".gmls       :", 'world_trade_vae_sian/' + str(i) + '.gml')
            print("sudo ./metadata < " + str(i) + ".gml")

            process = subprocess.Popen(["sudo ./metadata < world_trade_vae_sian/" + str(i) + ".gml"],
                                       stdout=subprocess.PIPE, shell=True)

            proc_stdout = process.communicate()[0].strip()

            with open("world_trade_vae_sian/" + str(i) + ".txt", 'wb') as fp:
                fp.write(proc_stdout)

            print("proc_stdout:", type(proc_stdout))
            print(" ")

    elif parliament is True:

        for i in range(n_repeats):

            # loading files
            # running the "./metadata" program

            process = subprocess.Popen(["sudo ./metadata < " "parliament.gml"],
                                       stdout=subprocess.PIPE, shell=True)

            proc_stdout = process.communicate()[0].strip()

            with open("parliament/SIAN_parliament_results" + str(i) + ".txt", 'wb') as fp: 
                fp.write(proc_stdout)

            print("proc_stdout:", type(proc_stdout))
            print(" ")

    elif parliament_vae is True:

        # loading files
        GMLs = glob.glob('parliament_vae_sian/*.gml')

        print("len(GMLs):", len(GMLs))

        # running the "./metadata" program
        for i in range(0, len(GMLs)):
            print(".gmls       :", 'parliament_vae_sian/' + str(i) + '.gml')
            print("sudo ./metadata < " + str(i) + ".gml")

            process = subprocess.Popen(["sudo ./metadata < parliament_vae_sian/" + str(i) + ".gml"],
                                       stdout=subprocess.PIPE, shell=True)

            proc_stdout = process.communicate()[0].strip()

            with open("parliament_vae_sian/" + str(i) + ".txt", 'wb') as fp:
                fp.write(proc_stdout)

            print("proc_stdout:", type(proc_stdout))
            print(" ")

    elif hvr is True:

        for i in range(n_repeats):
            # loading files
            # running the "./metadata" program
            process = subprocess.Popen(["sudo ./metadata < " "hvr.gml"],
                                       stdout=subprocess.PIPE, shell=True)

            proc_stdout = process.communicate()[0].strip()

            with open("hvr/SIAN_hvr_results" + str(i) + ".txt", 'wb') as fp:
                fp.write(proc_stdout)

            print("proc_stdout:", type(proc_stdout))
            print(" ")

    elif hvr_vae is True:

        # loading files
        GMLs = glob.glob('hvr_vae_sian/*.gml')

        print("len(GMLs):", len(GMLs))

        # running the "./metadata" program
        for i in range(0, len(GMLs)):
            print(".gmls       :", 'hvr_vae_sian/' + str(i) + '.gml')
            print("sudo ./metadata < " + str(i) + ".gml")

            process = subprocess.Popen(["sudo ./metadata < hvr_vae_sian/" + str(i) + ".gml"],
                                       stdout=subprocess.PIPE, shell=True)

            proc_stdout = process.communicate()[0].strip()

            with open("hvr_vae_sian/" + str(i) + ".txt", 'wb') as fp:
                fp.write(proc_stdout)

            print("proc_stdout:", type(proc_stdout))
            print(" ")

    elif cora_vae is True:

        # loading files
        GMLs = glob.glob('cora_vae_sian/*.gml')

        print("len(GMLs):", len(GMLs))

        # running the "./metadata" program
        for i in range(0, len(GMLs)):
            print(".gmls       :", 'cora_vae_sian/' + str(i) + '.gml')
            print("sudo ./metadata < " + str(i) + ".gml")

            process = subprocess.Popen(["sudo ./metadata < cora_vae_sian/" + str(i) + ".gml"],
                                       stdout=subprocess.PIPE, shell=True)

            proc_stdout = process.communicate()[0].strip()

            with open("cora_vae_sian/" + str(i) + ".txt", 'wb') as fp:
                fp.write(proc_stdout)

            print("proc_stdout:", type(proc_stdout))
            print(" ")

    elif facebook_0_vae is True:

        # loading files
        GMLs = glob.glob('facebook_0_vae_sian/*.gml')

        print("len(GMLs):", len(GMLs))

        # running the "./metadata" program
        for i in range(0, len(GMLs)):
            print(".gmls       :", 'facebook_0_vae_sian/' + str(i) + '.gml')
            print("sudo ./metadata < " + str(i) + ".gml")

            process = subprocess.Popen(["sudo ./metadata < facebook_0_vae_sian/" + str(i) + ".gml"],
                                       stdout=subprocess.PIPE, shell=True)

            proc_stdout = process.communicate()[0].strip()

            with open("facebook_0_vae_sian/" + str(i) + ".txt", 'wb') as fp:
                fp.write(proc_stdout)

            print("proc_stdout:", type(proc_stdout))
            print(" ")