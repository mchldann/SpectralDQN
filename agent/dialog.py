import matplotlib.pyplot as plt
import numpy as np

class Dialog(object):

    def __init__(self):
          self.plots = {}
          self.plot_width = 200
          self.first_update = True
          self.num_plots = None


    def add_data_point(self, plot_name, x_value, y_values, trim_x, show_graphs, log_dir):

        if plot_name not in self.plots:

            self.plots[plot_name] = {}

            self.plots[plot_name]["fig"] = plt.figure(num=plot_name)

            self.plots[plot_name]["fig"].canvas.draw()

            if show_graphs:
                plt.show(block=False)

            self.plots[plot_name]["x"] = np.empty([1, 1])

            self.num_plots = len(y_values)
            for i in range(0, self.num_plots):
                self.plots[plot_name]["y" + str(i)] = np.empty([1, 1])
                i += 1

            self.plots[plot_name]["min_y"] = y_values[0]
            self.plots[plot_name]["max_y"] = y_values[0] + 0.00001
            self.plots[plot_name]["trim_x"] = trim_x

        plot = self.plots[plot_name]
        plot["x"] = np.append(plot["x"], x_value)

        for i in range(0, self.num_plots):
            plot["y" + str(i)] = np.append(plot["y" + str(i)], y_values[i])

        if log_dir is not None:
            with open(log_dir + 'plot_' + plot_name + '.csv','a') as fd:
                fd.write(str(x_value) + ',' + str(y_values[0]) + '\n')


    def update_image(self, debug_txt):

        for key in self.plots:

            plt.figure(num=key)

            plot = self.plots[key]

            #plot["min_y"] = np.minimum(0.0, np.amin(plot["y0"]))
            #plot["max_y"] = np.amax(plot["y0"])

            min_y = 0.0
            max_y = np.NINF
            for i in range(0, self.num_plots):
                min_y = np.minimum(min_y, np.amin(plot["y" + str(i)]))
                max_y = np.maximum(max_y, np.amax(plot["y" + str(i)]))

            plot["min_y"] = min_y
            plot["max_y"] = max_y

            if plot["trim_x"] and len(plot["x"]) > self.plot_width:

                start_idx = len(plot["x"]) - self.plot_width
                end_idx = len(plot["x"]) - 1

                plot["x"] = plot["x"][start_idx:end_idx]
                for i in range(0, self.num_plots):
                    plot["y" + str(i)] = plot["y" + str(i)][start_idx:end_idx]

            self.plots[key]["fig"].canvas.set_window_title(key + ', ' + debug_txt)

            self.plots[key]["fig"].canvas.draw()

            # Just to avoid warnings about xmin = xmax
            if self.first_update:
                plt.xlim(plot["x"][0], plot["x"][0] + 0.00001)
                self.first_update = False
            else:
                plt.xlim(plot["x"][0], plot["x"][len(plot["x"]) - 1])

            plt.ylim(plot["min_y"], plot["max_y"])

            for i in range(0, self.num_plots):
                plt.plot(plot["x"], plot["y" + str(i)], c='C' + str(i % 10))

            plt.draw()
            plot["fig"].canvas.flush_events()

            plt.clf()


    def save_image(self, log_dir):

        for key in self.plots:

            plt.figure(num=key)

            plot = self.plots[key]

            #plot["min_y"] = np.minimum(0.0, np.amin(plot["y0"]))
            #plot["max_y"] = np.amax(plot["y0"])

            min_y = 0.0
            max_y = np.NINF
            for i in range(0, self.num_plots):
                min_y = np.minimum(min_y, np.amin(plot["y" + str(i)]))
                max_y = np.maximum(max_y, np.amax(plot["y" + str(i)]))

            plot["min_y"] = min_y
            plot["max_y"] = max_y

            self.plots[key]["fig"].canvas.draw()

            # Just to avoid warnings about xmin = xmax
            if self.first_update:
                plt.xlim(plot["x"][0], plot["x"][0] + 0.00001)
                self.first_update = False
            else:
                plt.xlim(plot["x"][0], plot["x"][len(plot["x"]) - 1])

            plt.ylim(plot["min_y"], plot["max_y"])

            for i in range(0, self.num_plots):
                plt.plot(plot["x"], plot["y" + str(i)], c='C' + str(i % 10))

            plt.draw()
            plot["fig"].canvas.flush_events()
            plot["fig"].savefig(log_dir + 'plot_' + str(key) + '.png', format='png')
            plt.clf()

