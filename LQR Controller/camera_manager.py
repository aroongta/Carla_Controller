'''
The camera manager sensor, it can be directly imported into any existing script. 
The camera can be attached to any actor or player. The camera view can be rendered and
displayed on a pygame window as well as stored as RGB images for each frame.
'''
# ---------Camera Manager
class CameraManager(object):
    def __init__(self,parent_actor):
        self.sensor=None
        self._surface=None
        self._parent=parent_actor
        self._recording=False
        self._camera_transforms=[carla.Transform(carla.Location(x=-5.5,z=2.8),carla.Rotation(pitch=-15)),
        carla.Transform(carla.Location(x=1.6,z=1.7))]
        self._transform_index=1
        self._sensors=[['sensor.camera.rgb', cc.Raw, 'Camera RGB'],
        ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)'],
        ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)'],
        ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)'],
        ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)'],
        ['sensor.camera.semantic_segmentation', cc.CityScapesPalette, 'Camera Semantic Segmentation (CityScapes Palette)'],
        ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)']]
        world=self._parent.get_world()
        bp_library=world.get_blueprint_library()
        for item in self._sensors:
            bp=bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                bp.set_attribute('image_size_x',str(1280))
                bp.set_attribute('image_size_y',str(720))
            elif item[0].startswith('sensor.lidar'):
                bp.set_attribute('range','5000')
            item.append(bp)
        self._index=None

    def toggle_camera(self):
        self._transform_index=(self._transform_index+1)%len(self._camera_transforms)
        self.sensor.set_transform(self._camera_transforms[self._transform_index])

    def set_sensor(self, index, notify=True):
        index = index % len(self._sensors)
        needs_respawn = True if self._index is None \
            else self._sensors[index][0] != self._sensors[self._index][0]
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self._surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self._sensors[index][-1],
                self._camera_transforms[self._transform_index],
                attach_to=self._parent)
            # We need to pass the lambda a weak reference to self to avoid
            # circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))

        self._index = index

    def next_sensor(self):
        self.set_sensor(self._index + 1)

    def toggle_recording(self):
        self._recording = not self._recording

    def render(self, display):
        if self._surface is not None:
            display.blit(self._surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        if self._sensors[self._index][0].startswith('sensor.lidar'):
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0]/3), 3))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self._hud.dim) / 100.0
            lidar_data += (0.5 * self._hud.dim[0], 0.5 * self._hud.dim[1])
            lidar_data = np.fabs(lidar_data)
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self._hud.dim[0], self._hud.dim[1], 3)
            lidar_img = np.zeros(lidar_img_size)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            # self._surface = pygame.surfarray.make_surface(lidar_img)
        else:
            image.convert(self._sensors[self._index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            # self._surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if self._recording:
            image.save_to_disk('_out/%08d' % image.frame_number)
