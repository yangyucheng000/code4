import os
import json
import argparse
import math
import grpc
import GrabSim_pb2_grpc
import GrabSim_pb2


def generate_data(args):
    channel = grpc.insecure_channel('localhost:30001', options=[
        ('grpc.max_send_message_length', 1024 * 1024 * 1024),
        ('grpc.max_receive_message_length', 1024 * 1024 * 1024)
    ])

    stub = GrabSim_pb2_grpc.GrabSimStub(channel)

    # DO NOT repeat init
    print('Please make sure that stub.Init is called only once, otherwise restart the simulator!!!')
    if args.version == 'v1':
        initworld = stub.Init(GrabSim_pb2.Count(value=1))
    else:
        initworld = stub.Init(GrabSim_pb2.NUL())
        initworld = stub.SetWorld(GrabSim_pb2.BatchMap(count=1, mapID=args.map_id))
    msg = stub.Reset(GrabSim_pb2.ResetParams())

    scene = stub.Observe(GrabSim_pb2.SceneID(value=0))

    with open('data_preprocess/name_replace_dict.json', 'r') as f:
        replace_name = json.load(f)
    if args.scene == 'TG':
        with open('data_preprocess/TG_objectname_replaced.json', 'r') as f:
            replace_obj_name = json.load(f)
    else:
        replace_obj_name = None

    if replace_obj_name is not None:
        for k in replace_obj_name:
            if k in replace_name:
                replace_name[k] = replace_obj_name[k]

    # origin_name = {}
    # for k in replace_name:
    #     origin_name[replace_name[k]] = k

    obj_list = []
    caption_list = []
    cat_ids = {}
    id = 0
    for i in range(len(scene.objects)):
        object = scene.objects[i]
        name = object.name.strip()
        if name in ['__UNKNOWN__', 'Ginger', 'Floor', 'Roof', 'Wall', 'Hand', 'Tongs']:
            continue
        if 'Room' not in name:
            if name in replace_name:
                new_name = replace_name[name]
            elif replace_obj_name is not None:
                new_name = replace_obj_name[name]
            else:
                new_name = name
            if new_name not in obj_list:
                obj_list.append(new_name)

                # if '-' in name:
                #     new_arr = name.split('-')
                #     new_obj = ' '.join(new_arr[1:]) + ' ' + new_arr[0]
                #     caption_list.append(new_obj.lower())
                if '(' in new_name:
                    new_arr = new_name.split('(')
                    new_obj = new_arr[-1].replace(')', '') + ' ' + new_arr[0].strip()
                    caption_list.append(new_obj.lower())
                else:
                    caption_list.append(new_name.lower())

                cat_ids[new_name] = id
                id += 1
    print(obj_list)
    print(caption_list)
    print(cat_ids)

    msg = stub.Reset(GrabSim_pb2.ResetParams())

    if args.scene == 'Starbucks':
        start_position = [-180, 140.0, 0, -1, 100]
    elif args.scene == 'TG':
        start_position = [0, -1350.0, 180, -1, 100]
    elif args.scene == 'NursingRoom':
        start_position = [-1369, 128, 0, -1, 100]
    else:
        start_position = None

    dataset = []
    eps_id = 0
    # delete_obj_num = 0
    for name in obj_list:

        shortest_distance = -1
        best_x, best_y, best_z = 0, 0, 0
        boundary_position = (0, 0)
        item = {}
        item['episode_id'] = eps_id
        eps_id += 1
        item['scene'] = args.scene

        if start_position is not None:
            msg = stub.Reset(GrabSim_pb2.ResetParams())
            msg = stub.Do(GrabSim_pb2.Action(
                sceneID=0,
                action=GrabSim_pb2.Action.ActionType.WalkTo,
                values=start_position
            ))
            assert msg.info not in ['Unreachable', 'Failed', 'AlreadyAtGoal']

        g_x, g_y, yaw = msg.location.X, msg.location.Y, msg.rotation.Yaw
        item['start_position'] = [g_x, g_y, yaw]
        item['goal'] = {}
        item['goal']['name'] = name
        item['goal']['cat_id'] = cat_ids[name]
        # delete_cat = False

        for i in range(len(scene.objects)):
            object = scene.objects[i]
            o_x, o_y, o_z = object.location.X, object.location.Y, object.location.Z
            if object.name in ['__UNKNOWN__', 'Ginger', 'Floor', 'Roof', 'Wall', 'Hand', 'Tongs']:
                continue
            if 'Room' in name:
                continue

            same_cat = False
            if replace_obj_name is None:
                if object.name in replace_name and name == replace_name[object.name]:
                    same_cat = True
                elif name == object.name:
                    same_cat = True
            elif name == replace_obj_name[object.name]:
                    same_cat = True


            if same_cat:
                # l2_distance = math.sqrt((g_x - o_x) ** 2 + (g_y - o_y) ** 2)
                # if l2_distance < shortest_distance or shortest_distance == -1:
                #     shortest_distance = l2_distance
                #     best_x, best_y, best_z = o_x, o_y, o_z

                msg = stub.Do(GrabSim_pb2.Action(
                    sceneID=0,
                    action=GrabSim_pb2.Action.ActionType.WalkTo,
                    values=[o_x, o_y, 0, 0, 350]
                ))
                if msg.info == 'Unreachable':
                    # delete_cat = True
                    # break
                    print(object)
                    # delete_obj_num += 1
                    continue
                message = msg.info.split(';')
                distance = float(message[0][22:])
                if distance < shortest_distance or shortest_distance == -1:
                    shortest_distance = distance
                    best_x, best_y, best_z = o_x, o_y, o_z
                    boundary_position = list(map(float, message[1][6:].split('|')[-1].split(',')))

        # if delete_cat:
        #     print('delete category: ', name)
        #     continue
        item['goal']['best_position'] = [best_x, best_y, best_z]
        item['goal']['boundary_position'] = boundary_position
        # item['goal']['shortest_l2_distance'] = shortest_distance
        item['goal']['shortest_distance'] = shortest_distance
        print(item)
        print(math.sqrt((best_x - boundary_position[0]) ** 2 + (best_y - boundary_position[1]) ** 2))
        dataset.append(item)

    # print(delete_obj_num)
    print('dataset len: ', len(dataset))
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    with open(os.path.join(args.save_path, args.scene + '.json'), 'w') as f:
        json.dump(dataset, f)
    with open(os.path.join(args.save_path, args.scene + '_categories.json'), 'w') as f:
        json.dump(caption_list, f)

    message = stub.Capture(GrabSim_pb2.CameraList(sceneID=0, cameras=[
        GrabSim_pb2.CameraName.Head_Segment,  # segmentation camera
        GrabSim_pb2.CameraName.Head_Depth,
        GrabSim_pb2.CameraName.Head_Color
    ]))
    items = message.info.split(';')

    object_names = {}
    for item in items:
        key, value = item.split(':')
        if value in replace_name:
            new_name = replace_name[value]
        elif replace_obj_name is not None and value in replace_obj_name:
            new_name = replace_obj_name[value]
        else:
            new_name = value
        object_names[int(key)] = new_name

    # del object_names[0]
    # del object_names[252]
    # del object_names[253]
    # del object_names[254]
    # del object_names[255]
    print(object_names)

    with open(os.path.join(args.save_path, args.scene + '_seg_categories_id.json'), 'w') as f:
        json.dump(object_names, f)


def main():
    parser = argparse.ArgumentParser(description="Generate ObjectNav Dataset")
    parser.add_argument(
        "--simulator_path",
        default="/liangxiwen/navigation/simulator-0223/",
        help="path to simulator",
    )
    parser.add_argument(
        '--map_id', type=int, default=3,
        help="3: Starbucks; 4: TG; 5: NursingRoom"
    )
    parser.add_argument(
        "--scene",
        default="TG",
        help="scene type, [TG, XBK, YLY]",
    )
    parser.add_argument(
        "--save_path",
        default="/liangxiwen/navigation/simulator-0223/objectnav",
        help="path to save dataset",
    )
    parser.add_argument(
        '--version', type=str, default="v2",
        help="dataset version, v1: single process (0410); v2: newest")
    args = parser.parse_args()

    if args.map_id == 3:
        args.scene = 'Starbucks'
    elif args.map_id == 4:
        args.scene = 'TG'
    elif args.map_id == 5:
        args.scene = 'NursingRoom'

    generate_data(args)


if __name__ == '__main__':
    main()
