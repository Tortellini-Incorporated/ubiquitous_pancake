/************************************************************
 *
 * All credit to https://sotrh.github.io/learn-wgpu/
 *
 ************************************************************/

use winit::{
	event::*,
	event_loop::{EventLoop, ControlFlow},
	window::{Window, WindowBuilder},
};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
	position:   [f32; 3],
	tex_coords: [f32; 2],
}

impl Vertex {
	fn desc<'a>() -> wgpu::VertexBufferDescriptor<'a> {
		wgpu::VertexBufferDescriptor {
			stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
			step_mode: wgpu::InputStepMode::Vertex,
			attributes: &[
				wgpu::VertexAttributeDescriptor {
					offset: 0,
					shader_location: 0,
					format: wgpu::VertexFormat::Float3,
				},
				wgpu::VertexAttributeDescriptor {
					offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
					shader_location: 1,
					format: wgpu::VertexFormat::Float2,
				}
			]
		}
	}
}

const VERTICES: &[Vertex] = &[
	Vertex { position: [-0.5,  0.5, 0.0], tex_coords: [0.0, 0.0], },
	Vertex { position: [-0.5, -0.5, 0.0], tex_coords: [0.0, 1.0], },
	Vertex { position: [ 0.5, -0.5, 0.0], tex_coords: [1.0, 1.0], },
	Vertex { position: [ 0.5,  0.5, 0.0], tex_coords: [1.0, 0.0], },
];

const INDICES: &[u16] = &[
	0, 1, 2,
	0, 2, 3,
];

struct State {
	surface: wgpu::Surface,
	device: wgpu::Device,
	queue: wgpu::Queue,
	sc_desc: wgpu::SwapChainDescriptor,
	swap_chain: wgpu::SwapChain,
	size: winit::dpi::PhysicalSize<u32>,
	render_pipeline: wgpu::RenderPipeline,
	vertex_buffer: wgpu::Buffer,
	index_buffer: wgpu::Buffer,
	num_indices: u32,
	diffuse_bind_group: wgpu::BindGroup,
}

impl State {
	async fn new(window: &Window) -> Self {
		let size = window.inner_size();

		let instance = wgpu::Instance::new(wgpu::BackendBit::PRIMARY);
		let surface  = unsafe { instance.create_surface(window) };
		let adapter  = instance.request_adapter(
			&wgpu::RequestAdapterOptions {
				power_preference:   wgpu::PowerPreference::Default,
				compatible_surface: Some(&surface),
			},
		).await.unwrap();

		let (device, queue) = adapter.request_device(
			&wgpu::DeviceDescriptor {
				features:          wgpu::Features::empty(),
				limits:            wgpu::Limits::default(),
				shader_validation: true,
			},
			None,
		).await.unwrap();

		let sc_desc = wgpu::SwapChainDescriptor {
			usage:        wgpu::TextureUsage::OUTPUT_ATTACHMENT,
			format:       wgpu::TextureFormat::Bgra8UnormSrgb,
			width:        size.width,
			height:       size.height,
			present_mode: wgpu::PresentMode::Fifo,
		};
		let swap_chain = device.create_swap_chain(&surface, &sc_desc);

		let diffuse_bytes = include_bytes!("<your-image-here>.png");
		let diffuse_image = image::load_from_memory(diffuse_bytes).unwrap();
		let diffuse_rgba  = diffuse_image.as_rgba8().unwrap();

		use image::GenericImageView;
		let dimensions = diffuse_image.dimensions();

		let texture_size = wgpu::Extent3d {
			width:  dimensions.0,
			height: dimensions.1,
			depth:  1
		};
		let diffuse_texture = device.create_texture(
			&wgpu::TextureDescriptor {
				size: texture_size,
				mip_level_count: 1,
				sample_count: 1,
				dimension: wgpu::TextureDimension::D2,
				format: wgpu::TextureFormat::Rgba8UnormSrgb,
				usage: wgpu::TextureUsage::SAMPLED | wgpu::TextureUsage::COPY_DST,
				label: Some("diffuse_texture"),
			}
		);
		queue.write_texture(
			wgpu::TextureCopyView {
				texture: &diffuse_texture,
				mip_level: 0,
				origin: wgpu::Origin3d::ZERO,
			},
			diffuse_rgba,
			wgpu::TextureDataLayout {
				offset: 0,
				bytes_per_row: 4 * dimensions.0,
				rows_per_image: dimensions.1,
			},
			texture_size,
		);

		let diffuse_texture_view = diffuse_texture.create_view(&wgpu::TextureViewDescriptor::default());
		let diffuse_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
			address_mode_u: wgpu::AddressMode::ClampToEdge,
			address_mode_v: wgpu::AddressMode::ClampToEdge,
			address_mode_w: wgpu::AddressMode::ClampToEdge,
			mag_filter: wgpu::FilterMode::Linear,
			min_filter: wgpu::FilterMode::Nearest,
			mipmap_filter: wgpu::FilterMode::Nearest,
			..Default::default()
		});

		let texture_bind_group_layout = device.create_bind_group_layout(
			&wgpu::BindGroupLayoutDescriptor {
				entries: &[
					wgpu::BindGroupLayoutEntry {
						binding: 0,
						visibility: wgpu::ShaderStage::FRAGMENT,
						ty: wgpu::BindingType::SampledTexture {
							multisampled: false,
							dimension: wgpu::TextureViewDimension::D2,
							component_type: wgpu::TextureComponentType::Uint,
						},
						count: None,
					},
					wgpu::BindGroupLayoutEntry {
						binding: 1,
						visibility: wgpu::ShaderStage::FRAGMENT,
						ty: wgpu::BindingType::Sampler {
							comparison: false,
						},
						count: None,
					},
				],
				label: Some("texture_bind_group_layout"),
			}
		);
		let diffuse_bind_group = device.create_bind_group(
			&wgpu::BindGroupDescriptor {
				layout: &texture_bind_group_layout,
				entries: &[
					wgpu::BindGroupEntry {
						binding: 0,
						resource: wgpu::BindingResource::TextureView(&diffuse_texture_view),
					},
					wgpu::BindGroupEntry {
						binding: 1,
						resource: wgpu::BindingResource::Sampler(&diffuse_sampler),
					},
				],
				label: Some("diffuse_bind_group"),
			}
		);

		let vs_module = device.create_shader_module(wgpu::include_spirv!("shader.vert.spv"));
		let fs_module = device.create_shader_module(wgpu::include_spirv!("shader.frag.spv"));

		let render_pipeline_layout =
			device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
				label: Some("Render Pipeline Layout"),
				bind_group_layouts: &[&texture_bind_group_layout],
				push_constant_ranges: &[],
			});

		let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
			label: Some("Render Pipeline"),
			layout: Some(&render_pipeline_layout),
			vertex_stage: wgpu::ProgrammableStageDescriptor {
				module: &vs_module,
				entry_point: "main",
			},
			fragment_stage: Some(
				wgpu::ProgrammableStageDescriptor {
					module: &fs_module,
					entry_point: "main",
				}
			),
			rasterization_state: Some(
				wgpu::RasterizationStateDescriptor {
					front_face: wgpu::FrontFace::Ccw,
					cull_mode: wgpu::CullMode::Back,
					depth_bias: 0,
					depth_bias_slope_scale: 0.0,
					depth_bias_clamp: 0.0,
					clamp_depth: false,
				}
			),
			color_states: &[
				wgpu::ColorStateDescriptor {
					format: sc_desc.format,
					color_blend: wgpu::BlendDescriptor::REPLACE,
					alpha_blend: wgpu::BlendDescriptor::REPLACE,
					write_mask: wgpu::ColorWrite::ALL,
				},
			],
			primitive_topology: wgpu::PrimitiveTopology::TriangleList,
			depth_stencil_state: None,
			vertex_state: wgpu::VertexStateDescriptor {
				index_format: wgpu::IndexFormat::Uint16,
				vertex_buffers: &[
					Vertex::desc(),
				],
			},
			sample_count: 1,
			sample_mask: !0,
			alpha_to_coverage_enabled: false,
		});

		let vertex_buffer = device.create_buffer_init(
			&wgpu::util::BufferInitDescriptor {
				label: Some("Vertex Buffer"),
				contents: bytemuck::cast_slice(VERTICES),
				usage: wgpu::BufferUsage::VERTEX,
			}
		);
		let index_buffer = device.create_buffer_init(
			&wgpu::util::BufferInitDescriptor {
				label: Some("Index Buffer"),
				contents: bytemuck::cast_slice(INDICES),
				usage: wgpu::BufferUsage::INDEX,
			}
		);
		let num_indices = INDICES.len() as u32;

		Self {
			surface,
			device,
			queue,
			sc_desc,
			swap_chain,
			size,
			render_pipeline,
			vertex_buffer,
			index_buffer,
			num_indices,
			diffuse_bind_group,
		}
	}

	fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
		self.size           = new_size;
		self.sc_desc.width  = new_size.width;
		self.sc_desc.height = new_size.height;
		self.swap_chain     = self.device.create_swap_chain(&self.surface, &self.sc_desc);
	}

	fn input(&mut self, event: &WindowEvent) -> bool {
		false
	}

	fn update(&mut self) {

	}

	fn render(&mut self) -> Result<(), wgpu::SwapChainError> {
		let frame = self
			.swap_chain
			.get_current_frame()?
			.output;

		let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
			label: Some("Render Encoder"),
		});

		{
			let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
				color_attachments: &[
					wgpu::RenderPassColorAttachmentDescriptor {
						attachment: &frame.view,
						resolve_target: None,
						ops: wgpu::Operations {
							load: wgpu::LoadOp::Clear(wgpu::Color {
								r: 0.1,
								g: 0.2,
								b: 0.3,
								a: 1.0,
							}),
							store: true,
						}
					}
				],
				depth_stencil_attachment: None,
			});

			render_pass.set_pipeline(&self.render_pipeline);
			render_pass.set_bind_group(0, &self.diffuse_bind_group, &[]);
			render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
			render_pass.set_index_buffer(self.index_buffer.slice(..));
			render_pass.draw_indexed(0..self.num_indices, 0, 0..1);
		}

		self.queue.submit(std::iter::once(encoder.finish()));

		Ok(())
	}
}

fn main() {
	env_logger::init();
	let event_loop = EventLoop::new();
	let window = WindowBuilder::new()
		.build(&event_loop)
		.unwrap();

	use futures::executor::block_on;
	let mut state = block_on(State::new(&window));

	event_loop.run(move |event, _, control_flow| {
		match event {
			Event::RedrawRequested(_) => {
				state.update();
				match state.render() {
					Ok(_) => {}
					Err(wgpu::SwapChainError::Lost) => state.resize(state.size),
					Err(wgpu::SwapChainError::OutOfMemory) => *control_flow = ControlFlow::Exit,
					Err(e) => println!("{:?}", e),
				}
			}
			Event::MainEventsCleared => {
				window.request_redraw();
			}
			Event::WindowEvent {
				ref event,
				window_id,
			} if window_id == window.id() => if !state.input(event) {
				match event {
					WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
					WindowEvent::KeyboardInput {
						input,
						..
					} => {
						match input {
							KeyboardInput {
								state: ElementState::Pressed,
								virtual_keycode: Some(VirtualKeyCode::Escape),
								..
							} => *control_flow = ControlFlow::Exit,
							_ => {}
						}
					}
					WindowEvent::Resized(physical_size) => {
						state.resize(*physical_size);
					}
					WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
						state.resize(**new_inner_size);
					}
					_ => {}
				}
			}
			_ => {}
		}
	});
}

