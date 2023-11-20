import pygame
import os
from OpenGL.GL import *

def MTL(filepath, obj_name):
    contents = {}
    mtl = None
    filename = os.path.join(filepath, obj_name)
    for line in open(filename, "r"):
        if line.startswith('#'): continue
        values = line.split()
        if not values: continue
        if values[0] == 'newmtl':
            mtl = contents[values[1]] = {}
        elif mtl is None:
            raise ValueError("mtl file doesn't start with newmtl stmt")
        elif values[0] == 'map_Kd':
            # load the texture referred to by this declaration
            mtl[values[0]] = values[1]
            surf = pygame.image.load(os.path.join(filepath, mtl['map_Kd']))
            image = pygame.image.tostring(surf, 'RGBA', 1)
            ix, iy = surf.get_rect().size
            texid = mtl['texture_Kd'] = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, texid)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,
                GL_LINEAR)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, ix, iy, 0, GL_RGBA,
                GL_UNSIGNED_BYTE, image)
        else:
            mtl[values[0]] = list(map(float, values[1:]))
    return contents

class OBJ:
    def __init__(self, filepath, objname, swapyz=False, normalise=True):
        """Loads a Wavefront OBJ file. """
        self.vertices = []
        self.normals = []
        self.texcoords = []
        self.faces = []

        vert_totals = [0,0,0]
        max_vertices = [-float('inf'), -float('inf'), -float('inf')]
        min_vertices = [float('inf'), float('inf'), float('inf')]

        filename = os.path.join(filepath, objname)

        material = None
        for line in open(filename, "r"):
            if line.startswith('#'): continue
            values = line.split()
            if not values: continue
            if values[0] == 'v':
                v = list(map(float, values[1:4]))
                if swapyz:
                    v = v[0], v[2], v[1]
                for i, v_val in enumerate(v):
                    vert_totals[i] += v_val
                    if v_val > max_vertices[i]:
                        max_vertices[i] = v_val
                    elif v_val < min_vertices[i]:
                        min_vertices[i] = v_val
                self.vertices.append(v)
            elif values[0] == 'vn':
                v = list(map(float, values[1:4]))
                if swapyz:
                    v = v[0], v[2], v[1]
                self.normals.append(v)
            elif values[0] == 'vt':
                self.texcoords.append(list(map(float, values[1:3])))
            elif values[0] in ('usemtl', 'usemat'):
                material = values[1]
            elif values[0] == 'mtllib':
                self.mtl = MTL(filepath, values[1])
            elif values[0] == 'f':
                face = []
                texcoords = []
                norms = []
                for v in values[1:]:
                    w = v.split('/')
                    face.append(int(w[0]))
                    if len(w) >= 2 and len(w[1]) > 0:
                        texcoords.append(int(w[1]))
                    else:
                        texcoords.append(0)
                    if len(w) >= 3 and len(w[2]) > 0:
                        norms.append(int(w[2]))
                    else:
                        norms.append(0)
                self.faces.append((face, norms, texcoords, material))

        vert_avgs = [0,0,0]

        for i in range(3):
            vert_avgs[i] = vert_totals[i] / len(self.vertices)
        
        if normalise:
            for vertex in self.vertices:
                # X and Y of each vertex is subtracted from the average position of all vertices (hopefully in the middle of the object)
                vertex[0] -= vert_avgs[0]
                vertex[1] -= vert_avgs[1]

                # Z of each vertex is subtracted from the height of the object (meaning the )
                vertex[2] -= min_vertices[2]

        self.gl_list = glGenLists(1)
        glNewList(self.gl_list, GL_COMPILE)
        glEnable(GL_TEXTURE_2D)
        glFrontFace(GL_CCW)
        for face in self.faces:
            vertices, normals, texture_coords, material = face

            mtl = self.mtl[material]
            if 'texture_Kd' in mtl:
                # use diffuse texmap
                glBindTexture(GL_TEXTURE_2D, mtl['texture_Kd'])
            else:
                # just use diffuse colour
                glColor(*mtl['Kd'])

            glBegin(GL_POLYGON)
            for i in range(len(vertices)):
                if normals[i] > 0:
                    glNormal3fv(self.normals[normals[i] - 1])
                if texture_coords[i] > 0:
                    glTexCoord2fv(self.texcoords[texture_coords[i] - 1])

                #print(self.vertices[vertices[i] - 1])
                glVertex3fv(self.vertices[vertices[i] - 1])
            glEnd()
        glDisable(GL_TEXTURE_2D)
        glEndList()
