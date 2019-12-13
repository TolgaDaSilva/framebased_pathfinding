#!/usr/bin/env python3

from argparse import ArgumentParser

import vizdoom as vzd
import matplotlib.lines as mlines
from utility import *

DEFAULT_CONFIG = Path("scenarios/custom.cfg")

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
    sleep_time = 0.028 # vzd.DEFAULT_TICRATE  # = 0.028

    # fig = plt.figure(frameon=False)

    for i in range(episodes):
        print("Episode #" + str(i + 1))

        game.new_episode()

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


            # # !!!! WAITKEY IS NECESSARY !!!!
            if cv2.waitKey(int(sleep_time * 1000)) == ord('p'):
                if explore_only:
                    print("exit explore only mode...")
                else:
                    print("switching to explore only mode...")
                explore_only = not explore_only

            game.advance_action()

            if not explore_only and frame % 15 == 0:
                p1, p2 = get_fov(pos_x, pos_y, angle, cam_fov, 300)
                create_nodes(pos_x, pos_y, p1[0], p1[1], p2[0], p2[1])
                label(pos_x, pos_y, pos_z, state.sectors, state.labels)
                # plot_fov(pos_x, pos_y, p1, p2)
                # plot_intersections()
                save_data(screen)

        print("Episode finished!")
        print("************************")

    cv2.destroyAllWindows()
    game.close()
