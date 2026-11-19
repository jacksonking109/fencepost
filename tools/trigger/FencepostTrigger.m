#import <Cocoa/Cocoa.h>
#import <Carbon/Carbon.h>
#include "trigger_write.h"

static NSString *const DefaultHotkey = @"ctrl+opt+cmd+f";

static BOOL parse_key(NSString *key, UInt32 *code)
{
    static const UInt32 letters[] = {
        kVK_ANSI_A, kVK_ANSI_B, kVK_ANSI_C, kVK_ANSI_D, kVK_ANSI_E, kVK_ANSI_F,
        kVK_ANSI_G, kVK_ANSI_H, kVK_ANSI_I, kVK_ANSI_J, kVK_ANSI_K, kVK_ANSI_L,
        kVK_ANSI_M, kVK_ANSI_N, kVK_ANSI_O, kVK_ANSI_P, kVK_ANSI_Q, kVK_ANSI_R,
        kVK_ANSI_S, kVK_ANSI_T, kVK_ANSI_U, kVK_ANSI_V, kVK_ANSI_W, kVK_ANSI_X,
        kVK_ANSI_Y, kVK_ANSI_Z,
    };
    static const UInt32 digits[] = {
        kVK_ANSI_0, kVK_ANSI_1, kVK_ANSI_2, kVK_ANSI_3, kVK_ANSI_4,
        kVK_ANSI_5, kVK_ANSI_6, kVK_ANSI_7, kVK_ANSI_8, kVK_ANSI_9,
    };
    static const UInt32 function_keys[] = {
        kVK_F1, kVK_F2, kVK_F3, kVK_F4, kVK_F5, kVK_F6, kVK_F7, kVK_F8, kVK_F9,
        kVK_F10, kVK_F11, kVK_F12, kVK_F13, kVK_F14, kVK_F15, kVK_F16, kVK_F17,
        kVK_F18, kVK_F19,
    };
    if (key.length == 1) {
        unichar c = [key characterAtIndex:0];
        if (c >= 'a' && c <= 'z') { *code = letters[c - 'a']; return YES; }
        if (c >= '0' && c <= '9') { *code = digits[c - '0']; return YES; }
    }
    for (NSUInteger i = 0; i < sizeof(function_keys) / sizeof(*function_keys); i++) {
        if ([key isEqualToString:[NSString stringWithFormat:@"f%lu", i + 1]]) {
            *code = function_keys[i];
            return YES;
        }
    }
    return NO;
}

static BOOL parse_hotkey(NSString *hotkey, UInt32 *code, UInt32 *modifiers)
{
    NSArray<NSString *> *parts = [hotkey.lowercaseString componentsSeparatedByString:@"+"];
    if (!parse_key(parts.lastObject, code)) return NO;
    *modifiers = 0;
    for (NSUInteger i = 0; i + 1 < parts.count; i++) {
        NSString *part = parts[i];
        UInt32 modifier;
        if ([part isEqualToString:@"ctrl"]) modifier = controlKey;
        else if ([part isEqualToString:@"opt"]) modifier = optionKey;
        else if ([part isEqualToString:@"cmd"]) modifier = cmdKey;
        else if ([part isEqualToString:@"shift"]) modifier = shiftKey;
        else return NO;
        if (*modifiers & modifier) return NO;
        *modifiers |= modifier;
    }
    return YES;
}

@interface TriggerApp : NSObject <NSApplicationDelegate>
- (void)saveWithReason:(const char *)reason;
@end

static OSStatus handle_hotkey(EventHandlerCallRef next, EventRef event, void *context)
{
    [(__bridge TriggerApp *)context saveWithReason:"hotkey"];
    return noErr;
}

@implementation TriggerApp {
    NSString *_path;
    NSStatusItem *_status;
    NSTimer *_feedback;
    EventHandlerRef _handler;
    EventHotKeyRef _hotkey;
}

- (void)applicationDidFinishLaunching:(NSNotification *)notification
{
    NSUserDefaults *defaults = NSUserDefaults.standardUserDefaults;
    _path = [defaults stringForKey:@"triggerFile"];
    if (!_path.length) {
        char buffer[64];
        const char *path = trigger_default_path(buffer);
        _path = [NSFileManager.defaultManager stringWithFileSystemRepresentation:path length:strlen(path)];
    }

    UInt32 code, modifiers;
    id preference = [defaults objectForKey:@"hotkey"];
    NSString *hotkey = DefaultHotkey;
    parse_hotkey(hotkey, &code, &modifiers);
    if (preference) {
        UInt32 customCode, customModifiers;
        if ([preference isKindOfClass:NSString.class] && parse_hotkey(preference, &customCode, &customModifiers)) {
            hotkey = preference;
            code = customCode;
            modifiers = customModifiers;
        } else {
            NSLog(@"Invalid fencepost hotkey %@. Using %@.", preference, DefaultHotkey);
        }
    }

    EventTypeSpec event = {kEventClassKeyboard, kEventHotKeyPressed};
    OSStatus status = InstallApplicationEventHandler(handle_hotkey, 1, &event, (__bridge void *)self, &_handler);
    if (status == noErr) {
        EventHotKeyID identifier = {'Fnce', 1};
        status = RegisterEventHotKey(code, modifiers, identifier, GetApplicationEventTarget(),
                                     kEventHotKeyExclusive, &_hotkey);
    }

    _status = [NSStatusBar.systemStatusBar statusItemWithLength:NSVariableStatusItemLength];
    _status.button.title = @"FP";
    _status.button.toolTip = @"Fencepost Trigger";
    NSMenu *menu = [[NSMenu alloc] init];
    menu.autoenablesItems = NO;
    NSString *title = [NSString stringWithFormat:@"Save timeline now (%@)", hotkey.uppercaseString];
    NSMenuItem *save = [menu addItemWithTitle:title action:@selector(saveFromMenu:) keyEquivalent:@""];
    save.target = self;
    if (status != noErr) {
        NSLog(@"Could not register fencepost hotkey %@ (%d)", hotkey, (int)status);
        [menu addItemWithTitle:@"Global hotkey unavailable" action:NULL keyEquivalent:@""].enabled = NO;
    }
    [menu addItemWithTitle:_path action:NULL keyEquivalent:@""].enabled = NO;
    [menu addItem:NSMenuItem.separatorItem];
    NSMenuItem *quit = [menu addItemWithTitle:@"Quit" action:@selector(terminate:) keyEquivalent:@""];
    quit.target = NSApp;
    _status.menu = menu;
}

- (void)saveFromMenu:(id)sender
{
    [self saveWithReason:"menu"];
}

- (void)saveWithReason:(const char *)reason
{
    const char *path = _path.fileSystemRepresentation;
    BOOL saved = path && trigger_write(path, reason) == 0;
    if (!saved) NSLog(@"Could not write fencepost trigger %@: %s", _path, strerror(path ? errno : EINVAL));
    _status.button.title = saved ? @"FP ✓" : @"FP !";
    [_feedback invalidate];
    _feedback = [NSTimer scheduledTimerWithTimeInterval:1.0 repeats:NO block:^(NSTimer *timer) {
        self->_status.button.title = @"FP";
    }];
}

- (void)applicationWillTerminate:(NSNotification *)notification
{
    if (_hotkey) UnregisterEventHotKey(_hotkey);
    if (_handler) RemoveEventHandler(_handler);
}
@end

int main(void)
{
    @autoreleasepool {
        NSApplication *app = NSApplication.sharedApplication;
        __attribute__((objc_precise_lifetime)) TriggerApp *delegate = [[TriggerApp alloc] init];
        app.delegate = delegate;
        [app run];
    }
    return 0;
}
