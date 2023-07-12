import os
import csv

base_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

out = open(os.path.join(base_path, "result", "combined_.csv"), "w")
out.write('Serial,#Dataset,#Module,Static,Update,Delta,Combination\n')
fileName = os.path.join(base_path, "result", "combined.csv")

ds_sum = {}
mod_sum = {}
with open(fileName) as csv_file:
    csv_reader = csv.reader(csv_file)
    next(csv_reader)

    for row in csv_reader:
        serial = row[0].strip()
        tcmb = row[1].strip()

        tcmb = tcmb.replace('(', '')
        tcmb = tcmb.split(')')

        no_ds = len(tcmb) - 1
        no_mod = 0
        for _c in tcmb[:-1]:
            tc = _c.split(':')
            tc = tc[1].split('-')[:-1]
            no_mod += len(tc)

        static = round(float(row[2].strip()) * 100.0, 2)
        update = round(float(row[3].strip()) * 100.0, 2)
        delta = round(float(row[4].strip()) * 100.0, 2)

        out.write(serial + ',' + str(no_ds) + ',' + str(no_mod) + ','
                  + str(static) + ',' + str(update) + ','
                  + str(delta) + ',' + row[1] + '\n')

        if no_ds not in ds_sum:
            ds_sum[no_ds] = (0, 0, 0)
        if no_mod not in mod_sum:
            mod_sum[no_mod] = (0, 0, 0)

        ds_sum[no_ds] = (ds_sum[no_ds][0] + 1,
                         ds_sum[no_ds][1] + static,
                         ds_sum[no_ds][2] + update)

        mod_sum[no_mod] = (mod_sum[no_mod][0] + 1,
                           mod_sum[no_mod][1] + static,
                           mod_sum[no_mod][2] + update)

out.close()

out = open(os.path.join(base_path, "result", "ds_stat.csv"), "w")
out.write('#Dataset,Static,Update,Delta\n')
for no_ds in ds_sum:
    static = ds_sum[no_ds][1] / ds_sum[no_ds][0]
    update = ds_sum[no_ds][2] / ds_sum[no_ds][0]

    out.write(str(no_ds) + ',' + str(static) + ',' + str(update) + ','
              + str(update - static) + '\n')
out.close()

out = open(os.path.join(base_path, "result", "mod_stat.csv"), "w")
out.write('#Module,Static,Update,Delta\n')
for no_ds in mod_sum:
    static = mod_sum[no_ds][1] / mod_sum[no_ds][0]
    update = mod_sum[no_ds][2] / mod_sum[no_ds][0]

    out.write(str(no_ds) + ',' + str(static) + ',' + str(update) + ','
              + str(update - static) + '\n')
out.close()
