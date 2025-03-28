def render_house():
    # Draw walls and rooms
    glBegin(GL_QUADS)
    # Living Room (Blue floor)
    glColor3f(0.2,0.2,0.8)
    glVertex3f(0,0,0)
    glVertex3f(6,0,0)
    glVertex3f(6,6,0)
    glVertex3f(0,6,0)

    # Kitchen (Yellow floor)
    glColor3f(0.8,0.8,0.2)
    glVertex3f(0,7,0)
    glVertex3f(6,7,0)
    glVertex3f(6,14,0)
    glVertex3f(0,14,0)

    # Draw furniture as cubes
    for obs in self.obstacles:
        glPushMatrix()
        glTranslatef(obs[0]+0.5, obs[1]+0.5, 0.5)
        glColor3f(0.5,0.2,0.1)
        glutSolidCube(1.0)
        glPopMatrix()

    # Draw doorways as arches
    for door in self.doorways:
        draw_arch(door[0], door[1])