"""Rebuild the tabletop scene from the preserved original floor scene."""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np


def build_tabletop(output=None):
    directory = Path(__file__).resolve().parent
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    root = ET.fromstring((directory/'scenes/floor-v1.xml').read_bytes(), parser=parser)
    root.set('model', 'Arm perception tabletop sandbox')
    root.find('statistic').set('center', '.12 0 .95')
    root.find('statistic').set('extent', '1.5')
    assets = root.find('asset')
    ET.SubElement(assets, 'material', name='table_wood', rgba='.63 .50 .35 1',
                  specular='.12', shininess='.15')
    ET.SubElement(assets, 'material', name='table_legs', rgba='.23 .28 .32 1',
                  specular='.25', shininess='.25')
    world = root.find('worldbody')
    table = ET.SubElement(world, 'body', name='table', pos='.12 0 .72')
    ET.SubElement(table, 'geom', name='tabletop', type='box', size='.80 .55 .04',
                  material='table_wood', group='0', friction='1.0 .01 .002')
    for index, (x, y) in enumerate(((-.69, -.44), (-.69, .44), (.69, -.44), (.69, .44))):
        ET.SubElement(table, 'geom', name=f'table_leg_{index}', type='box',
                      pos=f'{x} {y} -.38', size='.035 .035 .34', material='table_legs', group='0')
    mount = ET.SubElement(world, 'body', name='arm_mount', pos='-.26 .14 .76')
    for name in ('pedestal_foot', 'pedestal', 'base_ring', 'yaw_link'):
        child = next(item for item in world if item.get('name') == name)
        world.remove(child)
        mount.append(child)

    positions = ((.36, .26, .784), (.19, -.27, .796), (.55, -.03, .803))
    for index, position in enumerate(positions):
        body = ET.SubElement(world, 'body', name=f'table_object_{index}',
                              pos=' '.join(map(str, position)))
        ET.SubElement(body, 'joint', name=f'table_object_{index}_free', type='free',
                      damping='.03', armature='0', frictionloss='0')
        common = dict(group='2', friction='1.0 .01 .002', density='550')
        if index == 0:
            ET.SubElement(body, 'geom', name='table_object_0_core', type='capsule',
                          fromto='-.075 0 0 .075 0 0', size='.023', material='navy', **common)
            ET.SubElement(body, 'geom', name='table_object_0_stripe', type='box',
                          pos='0 -.024 0', size='.05 .003 .014', material='orange',
                          mass='0', contype='0', conaffinity='0', **common)
        elif index == 1:
            ET.SubElement(body, 'geom', name='table_object_1_core', type='box',
                          size='.055 .030 .035', material='navy', **common)
            ET.SubElement(body, 'geom', name='table_object_1_stripe', type='box',
                          pos='0 -.032 0', size='.033 .003 .021', material='orange',
                          mass='0', contype='0', conaffinity='0', **common)
        else:
            ET.SubElement(body, 'geom', name='table_object_2_core', type='sphere',
                          size='.042', material='orange', **common)
    # MuJoCo fills omitted keyframe qpos with zero, not the free bodies' XML pos.
    # Include complete object xyz/quaternion values so reset never drops them below the table.
    key = root.find('keyframe/key')
    key.set('qpos', key.get('qpos')+' '+' '.join(str(value) for position in positions
                                                for value in (*position, 1, 0, 0, 0)))
    camera = world.find("camera[@name='fixed']")
    # Keep the previous observer view for inspecting the scene; RGB perception
    # now uses an eye beside and above the shoulder on the fixed robot mount.
    ET.SubElement(world, 'camera', **{**camera.attrib, 'name': 'overview'})
    world.remove(camera)
    mount.append(camera)
    eye_offset = np.array([-.18, -.30, .80])
    eye = np.fromstring(mount.get('pos'), sep=' ')+eye_offset
    target = np.array([.28, 0, 1.08])
    forward = target-eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, 0, 1])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    camera.set('pos', ' '.join(f'{value:.8f}' for value in eye_offset))
    camera.set('xyaxes', ' '.join(f'{value:.8f}' for value in np.r_[right, up]))
    camera.set('fovy', '68')
    ET.indent(root, space='  ')
    output = Path(output) if output else directory/'scene.xml'
    output.write_text(ET.tostring(root, encoding='unicode')+'\n', encoding='utf-8')
    print(f'Built {output.resolve()}')


if __name__ == '__main__':
    build_tabletop()
