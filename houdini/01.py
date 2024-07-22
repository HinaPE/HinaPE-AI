import hapi

# 创建一个新的Houdini会话
session = hapi.create_in_process_session()

# 加载一个Houdini文件
hapi.load_hip_file(session, "example.hip")

# 运行Houdini中的操作
node_id = hapi.create_node(session, "SOP/box", "MyBox")
hapi.cook_node(session, node_id)

# 保存Houdini文件
hapi.save_hip_file(session, "example.hip")
