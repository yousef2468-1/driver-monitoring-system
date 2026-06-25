import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:socket_io_client/socket_io_client.dart' as io;
import 'package:audioplayers/audioplayers.dart';

const String SERVER_IP  = '10.0.0.213';
const String SERVER_URL = 'http://$SERVER_IP:5000';
const int    FPS_DELAY  = 67; // ~15 FPS

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final cameras = await availableCameras();
  runApp(DriverMonitorApp(cameras: cameras));
}

class DriverMonitorApp extends StatelessWidget {
  final List<CameraDescription> cameras;
  const DriverMonitorApp({super.key, required this.cameras});
  @override
  Widget build(BuildContext context) => MaterialApp(
    title: 'Driver Monitor',
    debugShowCheckedModeBanner: false,
    theme: ThemeData(colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xFF1F3864), brightness: Brightness.dark), useMaterial3: true),
    home: MonitorScreen(cameras: cameras),
  );
}

class MonitorScreen extends StatefulWidget {
  final List<CameraDescription> cameras;
  const MonitorScreen({super.key, required this.cameras});
  @override
  State<MonitorScreen> createState() => _MonitorScreenState();
}

class _MonitorScreenState extends State<MonitorScreen> with TickerProviderStateMixin {
  CameraController? _ctrl;
  bool _camReady  = false;
  int  _camIdx    = 1;
  bool _monitoring= false;
  bool _connected = false;
  bool _sending   = false;
  Timer? _timer;
  io.Socket? _socket;
  final AudioPlayer _audio = AudioPlayer();

  bool   _drowsy  = false;
  bool   _yawning = false;
  bool   _phone   = false;
  bool   _cig     = false;
  double _ear     = 0.0;
  double _mar     = 0.0;
  int    _yawns   = 0;
  double _score   = 100.0;
  String _grade   = 'A';
  String _color   = '00CC66';
  String _alert   = '';
  Uint8List? _img;

  int _dCount=0, _yCount=0, _pCount=0, _cCount=0;

  late AnimationController _anim;
  late Animation<double> _opacity;

  @override
  void initState() {
    super.initState();
    _anim = AnimationController(vsync: this, duration: const Duration(milliseconds: 400))..repeat(reverse: true);
    _opacity = Tween(begin: 0.4, end: 1.0).animate(_anim);
    _requestPerms();
  }

  Future<void> _requestPerms() async {
    await [Permission.camera, Permission.microphone].request();
    await _initCam();
  }

  Future<void> _initCam() async {
    if (widget.cameras.isEmpty) return;
    final idx = _camIdx < widget.cameras.length ? _camIdx : 0;
    _ctrl = CameraController(widget.cameras[idx], ResolutionPreset.low, enableAudio: false, imageFormatGroup: ImageFormatGroup.jpeg);
    try {
      await _ctrl!.initialize();
      if (mounted) setState(() => _camReady = true);
    } catch (e) { debugPrint('Cam: $e'); }
  }

  Future<void> _flipCam() async {
    _camIdx = _camIdx == 0 ? 1 : 0;
    await _ctrl?.dispose();
    setState(() => _camReady = false);
    await _initCam();
  }

  void _connectSocket() {
    _socket = io.io(SERVER_URL, io.OptionBuilder()
      .setTransports(['websocket'])
      .disableAutoConnect()
      .build());

    _socket!.onConnect((_) {
      setState(() => _connected = true);
      _startSending();
    });

    _socket!.onDisconnect((_) {
      setState(() { _connected = false; _monitoring = false; });
    });

    _socket!.on('result', (data) => _onResult(data));
    _socket!.connect();
  }

  void _startSending() {
    _timer = Timer.periodic(Duration(milliseconds: FPS_DELAY), (_) => _sendFrame());
  }

  Future<void> _sendFrame() async {
    if (!_monitoring || _sending || _ctrl == null || !_camReady) return;
    _sending = true;
    try {
      final XFile f  = await _ctrl!.takePicture();
      final bytes    = await f.readAsBytes();
      final b64      = base64Encode(bytes);
      _socket?.emit('frame', {'image': 'data:image/jpeg;base64,$b64'});
    } catch (_) {}
    _sending = false;
  }

  void _onResult(dynamic data) {
    if (!mounted) return;
    final drowsy = data['drowsy']    ?? false;
    final yawn   = data['yawning']   ?? false;
    final phone  = data['phone']     ?? false;
    final cig    = data['cigarette'] ?? false;
    final alerts = List<String>.from(data['alerts'] ?? []);

    // Play alert sound
    if (alerts.isNotEmpty) {
      _audio.play(AssetSource('alert.mp3')).catchError((_) {
        // fallback: system beep
      });
    }

    if (drowsy && !_drowsy) _dCount++;
    if (yawn   && !_yawning) _yCount++;
    if (phone  && !_phone)  _pCount++;
    if (cig    && !_cig)    _cCount++;

    Uint8List? img;
    final imgStr = data['image'] as String?;
    if (imgStr != null && imgStr.contains(',')) {
      img = base64Decode(imgStr.split(',')[1]);
    }

    setState(() {
      _drowsy=drowsy; _yawning=yawn; _phone=phone; _cig=cig;
      _ear=(data['ear']??0.0).toDouble(); _mar=(data['mar']??0.0).toDouble();
      _yawns=data['yawns']??0; _score=(data['score']??100.0).toDouble();
      _grade=data['grade']??'A'; _color=data['color']??'00CC66';
      _alert=alerts.isNotEmpty?alerts.first:''; _img=img;
    });
  }

  void _start() {
    setState(() { _monitoring=true; _score=100; _grade='A'; _dCount=0; _yCount=0; _pCount=0; _cCount=0; });
    _connectSocket();
  }

  void _stop() {
    _timer?.cancel();
    _socket?.disconnect();
    setState(() { _monitoring=false; _connected=false; _img=null; });
  }

  void _reset() {
    _socket?.emit('reset');
    setState(() { _score=100; _grade='A'; _dCount=0; _yCount=0; _pCount=0; _cCount=0; _alert=''; });
  }

  Color _hex(String h) { try { return Color(int.parse('FF$h',radix:16)); } catch(_){ return Colors.green; } }

  @override
  void dispose() { _timer?.cancel(); _socket?.dispose(); _ctrl?.dispose(); _anim.dispose(); _audio.dispose(); super.dispose(); }

  @override
  Widget build(BuildContext context) => Scaffold(
    backgroundColor: const Color(0xFF0A0A0A),
    body: SafeArea(child: Column(children: [
      _header(),
      _connBar(),
      if (_alert.isNotEmpty) _alertBanner(),
      Expanded(child: SingleChildScrollView(child: Column(children: [
        _feed(), _grid(), _metrics(), _controls(), _stats(),
        const SizedBox(height: 16),
      ]))),
    ])),
  );

  Widget _header() => Container(
    padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
    decoration: const BoxDecoration(gradient: LinearGradient(colors: [Color(0xFF1F3864),Color(0xFF2E75B6)])),
    child: Row(mainAxisAlignment: MainAxisAlignment.spaceBetween, children: [
      const Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Text('🚗 Driver Monitor', style: TextStyle(color: Colors.white, fontSize: 16, fontWeight: FontWeight.bold)),
        Text('Cairo University · FCAI · AI 2026', style: TextStyle(color: Colors.white70, fontSize: 11)),
      ]),
      Container(
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 6),
        decoration: BoxDecoration(color: _hex(_color), borderRadius: BorderRadius.circular(12)),
        child: Column(children: [
          Text(_score.toStringAsFixed(1), style: const TextStyle(color: Colors.white, fontSize: 22, fontWeight: FontWeight.w900)),
          Text('Grade $_grade', style: const TextStyle(color: Colors.white, fontSize: 11)),
        ]),
      ),
    ]),
  );

  Widget _connBar() => Container(
    color: const Color(0xFF1A1A1A), padding: const EdgeInsets.symmetric(vertical: 4),
    child: Row(mainAxisAlignment: MainAxisAlignment.center, children: [
      Container(width:8,height:8,decoration:BoxDecoration(shape:BoxShape.circle,color:_connected?Colors.green:Colors.red)),
      const SizedBox(width:6),
      Text(_connected?'🟢 Connected':'🔴 Not connected', style: const TextStyle(color:Colors.white70,fontSize:12)),
    ]),
  );

  Widget _alertBanner() => FadeTransition(
    opacity: _opacity,
    child: Container(width:double.infinity, color:Colors.red, padding:const EdgeInsets.symmetric(vertical:8),
      child: Text('⚠️ $_alert', textAlign:TextAlign.center, style:const TextStyle(color:Colors.white,fontSize:16,fontWeight:FontWeight.bold))),
  );

  Widget _feed() => Container(
    color: Colors.black, height: 300, width: double.infinity,
    child: _img != null
        ? Image.memory(_img!, fit: BoxFit.contain, gaplessPlayback: true)
        : _camReady && _ctrl != null
            ? CameraPreview(_ctrl!)
            : const Center(child: Text('📷 Ready', style: TextStyle(color:Colors.white))),
  );

  Widget _grid() => Padding(
    padding: const EdgeInsets.all(8),
    child: GridView.count(crossAxisCount:2, shrinkWrap:true, physics:const NeverScrollableScrollPhysics(),
      crossAxisSpacing:8, mainAxisSpacing:8, childAspectRatio:2.8,
      children: [
        _card('👁️','Drowsiness',_drowsy,_drowsy?'DROWSY!':'Alert'),
        _card('😮','Yawning',_yawning,_yawning?'YAWNING!':'Yawns: $_yawns'),
        _card('📱','Phone',_phone,_phone?'PHONE!':'No Phone'),
        _card('🚬','Smoking',_cig,_cig?'SMOKING!':'No Smoke'),
      ],
    ),
  );

  Widget _card(String icon, String label, bool active, String val) => AnimatedContainer(
    duration: const Duration(milliseconds:200),
    decoration: BoxDecoration(
      color: active?const Color(0xFF2E0D0D):const Color(0xFF0D2E1A),
      borderRadius: BorderRadius.circular(12),
      border: Border.all(color:active?Colors.red:Colors.green, width:active?2:1.5),
      boxShadow: active?[BoxShadow(color:Colors.red.withOpacity(0.3),blurRadius:8)]:[],
    ),
    child: Padding(padding:const EdgeInsets.symmetric(horizontal:10,vertical:6),
      child: Row(children: [
        Text(icon, style:const TextStyle(fontSize:22)),
        const SizedBox(width:8),
        Column(crossAxisAlignment:CrossAxisAlignment.start, children: [
          Text(label, style:TextStyle(color:Colors.white.withOpacity(0.7),fontSize:11)),
          Text(val, style:TextStyle(color:active?Colors.red:Colors.green,fontSize:13,fontWeight:FontWeight.bold)),
        ]),
      ]),
    ),
  );

  Widget _metrics() => Padding(
    padding: const EdgeInsets.symmetric(horizontal:8),
    child: Row(children: [
      _mbox('EAR',_ear.toStringAsFixed(3)), const SizedBox(width:8),
      _mbox('MAR',_mar.toStringAsFixed(3)), const SizedBox(width:8),
      _mbox('Yawns',_yawns.toString()),
    ]),
  );

  Widget _mbox(String l, String v) => Expanded(
    child: Container(padding:const EdgeInsets.symmetric(vertical:10),
      decoration:BoxDecoration(color:const Color(0xFF1A1A2E),borderRadius:BorderRadius.circular(10)),
      child:Column(children:[
        Text(v,style:const TextStyle(color:Color(0xFF4FC3F7),fontSize:18,fontWeight:FontWeight.w800)),
        Text(l,style:TextStyle(color:Colors.white.withOpacity(0.6),fontSize:11)),
      ]),
    ),
  );

  Widget _controls() => Padding(
    padding: const EdgeInsets.all(8),
    child: Row(children: [
      Expanded(child:_btn('▶️ Start',Colors.green,_monitoring?null:_start)),
      const SizedBox(width:8),
      Expanded(child:_btn('⏹ Stop',Colors.red,_monitoring?_stop:null)),
      const SizedBox(width:8),
      Expanded(child:_btn('🔄 Flip',Colors.blueGrey,_flipCam)),
      const SizedBox(width:8),
      Expanded(child:_btn('↺ Reset',Colors.indigo,_reset)),
    ]),
  );

  Widget _btn(String l, Color c, VoidCallback? fn) => GestureDetector(
    onTap:fn,
    child:AnimatedContainer(duration:const Duration(milliseconds:150),
      padding:const EdgeInsets.symmetric(vertical:12),
      decoration:BoxDecoration(color:fn!=null?c:c.withOpacity(0.3),borderRadius:BorderRadius.circular(12)),
      child:Text(l,textAlign:TextAlign.center,style:const TextStyle(color:Colors.white,fontSize:13,fontWeight:FontWeight.bold)),
    ),
  );

  Widget _stats() => Container(
    margin:const EdgeInsets.symmetric(horizontal:8),
    padding:const EdgeInsets.all(12),
    decoration:BoxDecoration(color:const Color(0xFF111111),borderRadius:BorderRadius.circular(12)),
    child:Column(crossAxisAlignment:CrossAxisAlignment.start,children:[
      const Text('📊 Session Stats',style:TextStyle(color:Colors.white70,fontSize:13,fontWeight:FontWeight.bold)),
      const SizedBox(height:8),
      Row(mainAxisAlignment:MainAxisAlignment.spaceAround,children:[
        _sitem('😴','Drowsy',_dCount), _sitem('😮','Yawns',_yCount),
        _sitem('📱','Phone',_pCount),  _sitem('🚬','Smoke',_cCount),
        _sitem('🏆','Score',_score.toInt()),
      ]),
    ]),
  );

  Widget _sitem(String i, String l, int v) => Column(children:[
    Text(i,style:const TextStyle(fontSize:18)),
    Text('$v',style:const TextStyle(color:Color(0xFF4FC3F7),fontSize:16,fontWeight:FontWeight.bold)),
    Text(l,style:TextStyle(color:Colors.white.withOpacity(0.5),fontSize:10)),
  ]);
}
