import DOME_communication as DOMEcomm
import DOME_transformation as DOMEtran

import numpy as np
import time
import cv2
from datetime import datetime

class ScreenManager():
    
    def __init__(self, output_dims, scale=1):
        self.output_dims = output_dims
        self.scale = scale
        self.pattern_dims = (output_dims[0]//scale, output_dims[1]//scale, 3)
        self.current = ''
        self.images = []
        self.screens = {}
        self.default_transformation = np.eye(3)
        self.store_image(np.zeros(self.pattern_dims))
        self.switch_to_screen('default')
    
    def set_scale(self, scale):
        old_scale = self.scale
        self.scale = scale
        self.pattern_dims = (self.output_dims[0]//scale, self.output_dims[1]//scale, 3)
        for i in range(len(self.images)):
            self.images[i] = cv2.resize(self.images[i], (self.pattern_dims[1],self.pattern_dims[0]))
        self.transform_shapes_on_screen(DOMEtran.linear_transform(scale=self.scale/old_scale))           
            
    def store_image(self, image):
        if image.shape == self.pattern_dims:
            self.images.append(image.astype(np.uint8))
        else:
            print(f'WARNING: image dimensions do not match screen' +
                  f'dimensions:\n--- image = {image.shape}\n' +
                  f'--- screen = {self.pattern_dims}')
    
    def set_image_to_screen(self, image_index):
        if image_index < len(self.images):
            self.screens[self.current]['image'] = image_index
        else:
            #images_list = [i for i in self.images.keys()]
            print(f'WARNING: image index {image_index} exceeds ' + \
                  f'number of stored images ({len(self.images)})')
    
    def switch_to_screen(self, new_screen):
        if new_screen == 'new':
            new_screen='screen' + str(len(self.screens.keys()))
        
        if not new_screen in self.screens.keys():
            poses = DOMEtran.PoseManager()
            self.screens[new_screen] = {'image': 0,
                                        'poses': poses,
                                        'shown': []}
        self.current = new_screen
    
    def make_gradient(self, points, values):
        x = np.arange(0, self.pattern_dims[1])
        v = np.ndarray([len(x),len(values[0])])
        for i in range(len(values[0])):
            v[:,i] = np.interp(x, points, [val[i] for val in values])
        pattern = np.multiply.outer(np.ones(self.pattern_dims[0]), v)
        return pattern.astype(np.uint8)
        
    def shapes_shown_to_screen(self, shown_list):
        self.screens[self.current]['shown'] = shown_list
    
    def add_shape_to_screen(self, label, shape_type, pose, colour):
        self.screens[self.current]['poses'].add_shape(label,
                                                      shape_type,
                                                      pose, colour)
        if not label in self.screens[self.current]['shown']:
            self.screens[self.current]['shown'].append(label)
    
    def transform_shapes_on_screen(self, matrix, labels=None):
        self.screens[self.current]['poses'].apply_transform(matrix,
                                                            labels)
    
    def set_colour_on_screen(self, colour, label, indices=None):
        self.screens[self.current]['poses'].set_colour(colour,
                                                       label,
                                                       indicies)
    
    def get_pattern_for_screen(self, axis_offset=None, labels=None):
        image_name = self.screens[self.current]['image']
        background = self.images[image_name]
        pose_manager = self.screens[self.current]['poses']
        display = pose_manager.draw_shapes(background, axis_offset,
                                           labels)
        return display

    def make_pattern_from_cmd(self, message):
        out_msg = 'Done'
        pattern = None
        
        # if the message is an array scale it to the right size
        # and use it as the pattern
        if isinstance(message, np.ndarray):
            image = message.copy().astype(np.uint8)
            if len(image.shape) == 1:
                image = (np.ones(self.pattern_dims)*image).astype(np.uint8)
            if image.shape[0:2] != self.pattern_dims[0:2]:
                image = cv2.resize(image, (self.pattern_dims[1],self.pattern_dims[0]) )
            # save and set the background for the current screen
            self.store_image(image)
            self.set_image_to_screen(len(self.images)-1)
            # make pattern from the saved screen data
            pattern = self.get_pattern_for_screen()
            
        # if the message is a dictionary extract the commands from it
        #   message = {'screen': 'default',
        #              'image': image_index,
        #              'add': {'label': 'spotlight', 'shape': 'circle',
        #                      'pose': [[1,0,0],[0,1,0],[0,0,1]], 'colour':[255]},
        #              'add2': {'label': 'spotlight', 'shape': 'circle',
        #                       'pose': [[1,0,0],[0,1,0],[0,0,1]], 'colour':[255]},
        #              'transform': {'matrix': [[1,0,0],[0,1,0],[0,0,1]], 'labels': ['a']},
        #              'transform2': {'matrix': [[1,0,0],[0,1,0],[0,0,1]], 'labels': ['b']},
        #              'colour': {'colour': [255], 'label': 'spotlight', 'indices': [0]},
        #              'shown': ['name1', 'name2']}
        elif isinstance(message, dict):
            all_command_types = ['get', 'set', 'scale', 'gradient',
                                 'screen', 'image', 'add',
                                 'transform', 'colour', 'shown']
            if not all([any([command_type in c for command_type in all_command_types])
                        for c in message.keys()]):
                out_msg=f'Unrecognized command!'
            
            for command_type in all_command_types:
                commands = [c for c in message.keys()
                            if command_type in c]
                
                for c in commands:
                    # message = {"get": {'param': name_of_the_attribute}}                        
                    if command_type == 'get':
                        if hasattr(self, message[c]['param']):
                            val = getattr(self, message[c]['param'])
                            if isinstance(val, np.ndarray):
                                out_msg = val.tolist()
                            else:
                                out_msg = val
                            print(f'{message[c]["param"]} = {out_msg}')
                        else:
                            out_msg = f'{message[c]["param"]} is not valid!'
                    
                    # message = {"set": {'param': name_of_the_attribute , 'value': new_value}}                        
                    elif command_type == 'set':
                        if hasattr(self, message[c]['param']):
                            old_val = getattr(self, message[c]['param'])
                            if isinstance(old_val, np.ndarray):
                                new_val = np.array(message[c]['value'])
                            else:
                                new_val = message[c]['value']
                            setattr(self, message[c]['param'], new_val)
                            print(f'{message[c]["param"]} = {getattr(self, message[c]["param"])}')
                        else:
                            out_msg = f'{message[c]["param"]} is not valid!'
                    
                    # message = {"scale": new_value}                        
                    elif command_type == 'scale':
                        self.set_scale(message[c])
                    
                    # message = {"gradient": {'points': [x1, x2], 'values': [light1, light2]}}                        
                    elif command_type == 'gradient':
                        points = message[c]['points']
                        points = [p/self.scale for p in points]
                        values = message[c]['values']
                        pattern = self.make_gradient(points, values)
                        self.store_image(pattern)
                        self.set_image_to_screen(len(self.images)-1)
                    
                    # message = {"screen": screen_name}                        
                    elif command_type == 'screen':
                        self.switch_to_screen(
                                message[c])
                    
                    # message = {'image': image_index}
                    elif command_type == 'image':
                        self.set_image_to_screen(
                                message[c])
                    
                    # message = {"add": {"label": 'a', "shape type": 'square',
                    #                   "pose": [[20, 0, 300], [0, 100, 500],[0, 0, 1]],
                    #                   "colour": [0, 255, 0]}}
                    elif command_type == 'add':
                        label = message[c]['label']
                        shape_type = message[c]['shape type']
                        pose = np.array(message[c]['pose'])
                        pose = np.dot(DOMEtran.linear_transform(scale=1/self.scale),pose)
                        colour = None
                        if 'colour' in message[c].keys():
                            colour = message[c]['colour']
                        self.add_shape_to_screen(
                                label, shape_type, pose, colour)
                    
                    # message = {'transform': {'matrix': [[1,0,0],[0,1,0],[0,0,1]], 'labels': ['a']}}
                    elif command_type == 'transform':
                        matrix = np.array(message[c]['matrix'])
                        matrix = np.dot(DOMEtran.linear_transform(scale=1/self.scale),matrix)
                        labels = None
                        if 'labels' in message[c].keys():
                            labels = message[c]['labels']
                        self.transform_shapes_on_screen(
                                matrix, labels)
                    
                    # message = {'colour': {'colour': [255], 'label': 'splotlight',
                    #                       'indices': [0]}}
                    elif command_type == 'colour':
                        colour = message[c]['colour']
                        label = message[c]['label']
                        indices = None
                        if 'indices' in message[c].keys():
                            indices = message[c]['indices']
                        self.set_colour_on_screen(
                                colour, label, indices)
                    
                    #  message = {"shown": list_of_labels_to_show}
                    elif command_type == 'shown':
                        self.shapes_shown_to_screen(
                                message[c])
            
            # make pattern from the saved screen data
            pattern = self.get_pattern_for_screen()

        # if the message is a string extract the command from it
        elif isinstance(message, str):
            pattern = np.zeros(self.pattern_dims)
            if message == 'exit':
                cv2.destroyAllWindows()
                out_msg='exit'
                time.sleep(1)
                projecting = False
            segments = message.split(' ')
            if segments[0] == 'dimensions':
                out_msg = self.output_dims
            elif segments[0] == 'all':
                for c in range(0, 3):
                    pattern[:, :, c] = int(segments[c + 1])
            elif segments[0] == 'row':
                pattern[int(segments[1]):int(segments[2]),
                        :, :] = 255
            elif segments[0] == 'column':
                pattern[:, int(segments[1]):int(segments[2]),
                        :] = 255
            else:
                out_msg=f'Unrecognised string:' + f'{message}'
        
        # if the message is of a different type throw an error
        else:
            out_msg='Unexpected data type!'
        
        if pattern is not None:
            pattern = cv2.resize(pattern, (self.output_dims[1],self.output_dims[0]))
        
        return pattern, out_msg

def main(output_dims, refresh_delay, scale=1):
    screen_manager = ScreenManager(output_dims, scale)
    pattern = screen_manager.get_pattern_for_screen()
    cv2.namedWindow('Pattern', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Pattern', cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN)
    
    with DOMEcomm.NetworkNode() as dome_pi0node:
        cv2.imshow('Pattern', pattern)
        cv2.waitKey(0)
        dome_pi0node.establish_connection()

        projecting = True
        while projecting:
            message = dome_pi0node.receive()
            tic=datetime.now()
            
            pattern, out_msg = screen_manager.make_pattern_from_cmd(message)
            
            if out_msg == 'Done':                
                # update projected pattern
                #ratio = (output_dims[0]/pattern.shape[0], output_dims[1]/pattern.shape[1])
                #pattern2output = DOMEtran.linear_transform(scale=ratio)
                #pattern2camera = np.dot(pattern2output,screen_manager.default_transformation)
                #out_pattern = DOMEtran.transform_image(pattern, screen_manager.default_transformation, output_dims)
                cv2.imshow('Pattern', pattern)
                cv2.waitKey(refresh_delay)
            
            toc=datetime.now()
            ellapsed_time = (toc - tic).total_seconds()
            print(f'{out_msg}')
            print(f'Update time = {ellapsed_time:4.3}s\n')
            dome_pi0node.transmit(out_msg)

if __name__ == '__main__':
    scale = 1
    output_dims = (480, 854, 3)
    
    refresh_delay = 33    # refresh delay in milliseconds
    
    main(output_dims, refresh_delay, scale)
