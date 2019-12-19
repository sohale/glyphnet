"""
pcolor: for plotting pcolor using matplotlib
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import time

def is_linux():
    import platform
    s = platform.system()
    return {
        'Linux': True,
        'Darwin': False,
        'Windows': False,
    }[s]

def is_mac():
    import platform
    s = platform.system()
    return {
        'Linux': False,
        'Darwin': True,
        'Windows': False,
    }[s]

def linux_plot_issue():
    if is_linux():
        import matplotlib
        matplotlib.use('TkAgg')
        # matplotlib.use('agg')
        print('backend:', matplotlib.get_backend())
        # matplotlib.hold(true) # deprecated


output_directory = './generated'
os.makedirs(output_directory, exist_ok=True)

class PColor:
    """ Show and save pcolor (w,h,3) in range float [0,1] """
    @staticmethod
    def plot_show_image(G_paintings2d, file_id, sleep_sec, more_info):
        plt.clf()

        #print(np.max(np.max(G_paintings2d,axis=2), axis=0))
        #print(np.min(np.min(G_paintings2d,axis=2), axis=1))
        #print(G_paintings2d.shape)
        #plt.imshow(G_paintings2d)
        #plt.imshow((G_paintings2d * 0.2 + 0.5)*0.2)
        #img_pix_rescale = (G_paintings2d * 0.05 + 0.5)
        #img_pix_rescale = (G_paintings2d)
        #plt.imshow(img_pix_rescale, vmin=-100, vmax=100)
        img_pix_rescale = ((G_paintings2d) / 80.0 *40  ) +0.5
        plt.imshow((img_pix_rescale *128).astype(np.uint8))
        print('min max:', np.min(img_pix_rescale.ravel()), np.max(img_pix_rescale.ravel()))
        #plt.pcolor(np.mean(G_paintings2d, axis=2))
        print("@*")
        acc, score = more_info
        plt.text(-.5, 0, 'D accuracy=%.2f (0.5 for D to converge)' % acc, fontdict={'size': 15})
        plt.text(-.5, G_paintings2d.shape[1]*0.5, 'D score= %.2f (-1.38 for G to converge)' % score, fontdict={'size': 15})
        plt.colorbar()


        PColor.next_plot(sleep_sec)

        if(file_id is not None):
            PColor.save( os.path.join(output_directory, file_id + '.png') )

    @staticmethod
    def save(filename):
        plt.draw()
        plt.savefig( filename )
        print("saved")
        if is_mac():
            wait_time_sec = 0.1
            time.sleep(wait_time_sec)

    """ Next plot. Platform-independent """
    @staticmethod
    def next_plot(sleep_sec):
        if is_mac():
            print('draw')
            import sys
            sys.stdout.flush()

            plt.draw()
            time.sleep(sleep_sec)
        elif is_linux():
            # """ "Modal" """
            # plt.show()


            #plt.draw()
            #plt.show(block=False)
            #time.sleep(0.5)
            #plt.draw()
            """
            # futile:
            plt.ion()
            plt.draw()
            plt.show()
            plt.ioff()

            time.sleep(sleep_sec)
            time.sleep(2.0)
            plt.close()
            plt.ioff()
            """
        else:
            raise

    @staticmethod
    def init():
        linux_plot_issue()
        print('matplotlib init.')
        plt.cla()
        #plt.imshow(main_artworks[0])

        if is_linux():
            # plt.ioff()  # not necessary
            # plt.show()

            #plt.ion()
            plt.draw()
            plt.show(block=False)
            plt.draw()
            time.sleep(0.5)
            return
        elif is_mac():
            plt.draw()
            plt.ion()
            plt.show()
            time.sleep(0.1)
            plt.ion()   # something about continuous plotting
            return
        else:
            raise
        raise

    @staticmethod
    def last(self):
        if is_mac():
            plt.ioff()
            plt.show()
        elif is_linux():
            pass
        else:
            raise
