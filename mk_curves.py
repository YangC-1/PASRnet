import matplotlib.pyplot as plt
import numpy as np
import os


def mk_curve(txt_path, model_names):
    legends = model_names
    fig_name = '_'.join(model_names) + '.png'
    save_dir = 'PASR_results/'

    psnrs = []
    for file in txtpath:
        f = np.loadtxt(file, dtype=float, delimiter=' ')
        psnrs.append(f)

    # make plots
    fig = plt.figure(figsize=(5, 4))
    line_style = ['k.--', 'r.--', 'b.--', 'g.--']
    for i, p in enumerate(psnrs):
        x = np.asarray(range(0, p.shape[0])) + 1
        plt.plot(x, p, line_style[i])
    plt.xlabel('Epochs')
    plt.ylabel('PSNR(dB)')
    plt.legend(legends)
    plt.show()

    # save fig
    save_dir = os.path.join(os.getcwd(), save_dir)
    os.makedirs(save_dir) if not os.path.isdir(save_dir) else print('Saving fig...')
    fig.savefig(os.path.join(save_dir, fig_name), dpi=300)


if __name__ == "__main__":
    # .txt to load
    txtpath = [os.path.join(os.getcwd(), 'PASR_results/densenetsr_scale4/densenetsr_psnrs.txt'),
               os.path.join(os.getcwd(), 'PASR_results/rdn_scale4/rdn_psnrs.txt'),
               os.path.join(os.getcwd(), 'PASR_results/resnetsr_scale4/resnetsr_psnrs.txt'),
               os.path.join(os.getcwd(), 'PASR_results/resunet_scale4/resunet_psnrs.txt')]
    model_name = ['PASRnetx4', 'RDNx4', 'SRResNetx4', 'ResUnetx4']
    mk_curve(txtpath, model_names=model_name)
