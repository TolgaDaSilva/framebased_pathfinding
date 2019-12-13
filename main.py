#!/usr/bin/env python3

from argparse import ArgumentParser

import vizdoom as vzd
import matplotlib.lines as mlines
from utility import *

## Used to generate train data, plot the labels  ##

DEFAULT_CONFIG = Path("scenarios/config.cfg")

if __name__ == "__main__":
    parser = ArgumentParser("ViZDoom example showing different buffers (screen, depth, labels).")
    parser.add_argument(dest="config",
                        default=str(DEFAULT_CONFIG),
                        nargs="?",
                        help="Path to the configuration file of the scenario."
                             " Please see "
                             "../../scenarios/*cfg for more scenarios.")

    args = parser.parse_args()

    explore_only = True

    game = vzd.DoomGame()

    # Use other config file if you wish.
    game.load_config(args.config)
    if not explore_only:
        game.set_doom_scenario_path("scenarios/" + SCENARIO_FILE)

    game.add_game_args('-nomonsters 1')
    # game.add_game_args('+freelook 0')

    game.init()
    episodes = 10
    sleep_time = 0.028

    fig = plt.figure(frameon=False)

    for i in range(episodes):
        print("Episode #" + str(i + 1))

        game.new_episode()

        # enable grid, size of the grid is 128 in map units, which maps to 3 meters.
        # game.send_game_command('am_grid 1')

        frame = 0

        while not game.is_episode_finished():
            frame += 1
            state = game.get_state()
            pos_x = game.get_game_variable(vzd.GameVariable.POSITION_X)
            pos_y = game.get_game_variable(vzd.GameVariable.POSITION_Y)
            pos_z = game.get_game_variable(vzd.GameVariable.POSITION_Z)
            angle = game.get_game_variable(vzd.GameVariable.ANGLE)
            pitch = game.get_game_variable(vzd.GameVariable.PITCH)

            cam_fov = game.get_game_variable(vzd.GameVariable.CAMERA_FOV)
            screen = state.screen_buffer
            cv2.imshow('ViZDoom Screen Buffer', screen)

            depth = state.depth_buffer
            # if depth is not None:
            #     cv2.imshow('ViZDoom Depth Buffer', depth)

            labels = state.labels_buffer
            # if labels is not None:
            #    cv2.imshow('ViZDoom Labels Buffer', segmentation(labels, pitch))

            automap = state.automap_buffer
            # if automap is not None:
            #     cv2.imshow('ViZDoom Map Buffer', automap)


            print_sectors(state.sectors)
            print_objects(state.objects, pos_x, pos_y, pos_z)

            # # !!!! WAITKEY IS NECESSARY !!!!
            if cv2.waitKey(int(sleep_time * 1000)) == ord('p'):
                print('writing images...')
                cv2.imwrite('autoMap.png', automap)
                cv2.imwrite('label.png', labels)
                cv2.imwrite('seg_label.png', segmentation(labels, pitch))
                cv2.imwrite('gray_label.png', labels)
                cv2.imwrite('depth.png', depth)
                cv2.imwrite('screen.png', screen)
                print('finished')

            game.advance_action()

            if frame % 20 == 0:
                p1, p2 = get_fov(pos_x, pos_y, angle, cam_fov, 300)
                create_nodes(pos_x, pos_y, p1[0], p1[1], p2[0], p2[1])
                label(pos_x, pos_y, pos_z, state.sectors, state.labels)
                print("P1: ", p1)
                print("POS X: ", pos_x)
                print("ABS : ", calc_distance(pos_x,pos_x,0,p1[0],p1[1],0))
                plot_fov(pos_x, pos_y, p1, p2)
                plot_intersections()
                if not explore_only:
                    save_data(screen)

            # Show map
            # plt.xlim(pos_x-224, pos_x+15)
            legend_walls = mlines.Line2D([],[], color='black', label='Wände')
            legend_obstacles = mlines.Line2D([],[], color='blue', label='Hindernisse')
            legend_player, = plt.plot(pos_x,pos_y, color='#000099',marker='o', label='Spieler')
            legend_free, = plt.plot([], [], marker='o', color='green', mew=0.1, label='begehbar')
            legend_blocked, = plt.plot([], [], marker='o', color='red', mew=0.1, label='blockiert')
            plt.axis('off')
            plt.legend(handles=[legend_walls,legend_obstacles,legend_player,legend_free,legend_blocked],loc=0, prop={'size': 14})
            plt.show()
            plt.close()

        print("Episode finished!")
        print("************************")

    cv2.destroyAllWindows()
    game.close()
